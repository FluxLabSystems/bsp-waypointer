// recast_c: a flat C ABI over Recast's solo-mesh build, loaded by bsp_waypointer.recast_native through ctypes.
//
// Coordinates: the caller passes Source's z-up world (x, y, z). Recast is y-up, so they are rotated in as
// (x, z, -y) -- a rotation, not a mirror, so triangle winding (and so the walkable-normal test) is kept --
// and rotated back out as (x, -z', y').
//
// Config (float array, in order; world units and degrees):
//   0 cell_size  1 cell_height  2 walkable_slope_deg  3 agent_height  4 agent_climb  5 agent_radius
//   6 max_edge_len  7 max_edge_error  8 min_region_area (cells)  9 merge_region_area (cells)  10 verts_per_poly
//   11 filter_ledges (0/1)
// Solid volumes: Recast rasterises surfaces, so the floor under a brush with no faces of its own (a player
// clip: vbsp never splits the floor's faces for it) stays walkable inside it. `planes` holds such brushes as
// convex sets of Source planes (nx, ny, nz, d: inside is n . p < d), `counts` how many planes each has; a
// walkable span whose floor (half a cell up) is inside one is made unwalkable before erosion.
// Heights and the climb are converted to cells by rounding DOWN for the climb (never climbs more than asked)
// and UP for the agent height and radius (never fits where the hull does not).

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "Recast.h"

#ifdef _WIN32
#define RCW_API extern "C" __declspec(dllexport)
#else
#define RCW_API extern "C"
#endif

struct rcw_result
{
	int nverts;          // polygon-mesh vertices
	float* verts;        // nverts * 3, Source world (x, y, z)
	int npolys;
	int nvp;             // max vertices per polygon
	int* polys;          // npolys * 2 * nvp: vertex indices then neighbour polygon indices, -1 = none
	int* regions;        // npolys
	int grid_w, grid_h;  // heightfield size in cells, for diagnostics
};

static char g_err[512];

static void set_err(const char* msg)
{
	snprintf(g_err, sizeof(g_err), "%s", msg);
}

RCW_API const char* rcw_last_error()
{
	return g_err;
}

RCW_API int rcw_version()
{
	return 2;
}

RCW_API void rcw_free(rcw_result* r)
{
	if (!r)
		return;
	free(r->verts);
	free(r->polys);
	free(r->regions);
	free(r);
}

struct Owned
{
	rcHeightfield* hf = 0;
	rcCompactHeightfield* chf = 0;
	rcContourSet* cset = 0;
	rcPolyMesh* pmesh = 0;
	float* rv = 0;
	unsigned char* areas = 0;
	~Owned()
	{
		rcFreeHeightField(hf);
		rcFreeCompactHeightfield(chf);
		rcFreeContourSet(cset);
		rcFreePolyMesh(pmesh);
		free(rv);
		free(areas);
	}
};

static bool inside_brush(const float* pl, int n, float x, float y, float z)
{
	for (int i = 0; i < n; ++i)
		if (pl[i * 4] * x + pl[i * 4 + 1] * y + pl[i * 4 + 2] * z >= pl[i * 4 + 3] - 0.01f)
			return false;
	return true;
}

static int carve_solids(rcCompactHeightfield& chf, const float* planes, const int* counts, int nbrushes)
{
	int carved = 0;
	const float* pl = planes;
	for (int b = 0; b < nbrushes; ++b)
	{
		const int n = counts[b];
		for (int cz = 0; cz < chf.height; ++cz)
		{
			for (int cx = 0; cx < chf.width; ++cx)
			{
				const rcCompactCell& c = chf.cells[cx + cz * chf.width];
				const float sx = chf.bmin[0] + (cx + 0.5f) * chf.cs;
				const float sy = -(chf.bmin[2] + (cz + 0.5f) * chf.cs);
				for (int i = (int)c.index, ni = (int)(c.index + c.count); i < ni; ++i)
				{
					if (chf.areas[i] == RC_NULL_AREA)
						continue;
					const float sz = chf.bmin[1] + (chf.spans[i].y + 0.5f) * chf.ch;
					if (inside_brush(pl, n, sx, sy, sz))
					{
						chf.areas[i] = RC_NULL_AREA;
						++carved;
					}
				}
			}
		}
		pl += n * 4;
	}
	return carved;
}

RCW_API int rcw_build(const float* verts, int nverts, const int* tris, int ntris, const float* cfg, int ncfg,
                      const float* planes, const int* counts, int nbrushes, rcw_result** out)
{
	*out = 0;
	g_err[0] = 0;
	if (ncfg < 12)
	{
		set_err("config needs 12 values");
		return 0;
	}
	if (nverts <= 0 || ntris <= 0)
	{
		set_err("empty mesh");
		return 0;
	}
	const float cs = cfg[0], ch = cfg[1];
	const int walkableHeight = (int)ceilf(cfg[3] / ch);
	const int walkableClimb = (int)floorf(cfg[4] / ch + 1e-4f);
	const int walkableRadius = (int)ceilf(cfg[5] / cs - 1e-4f);
	const int nvp = (int)cfg[10];

	rcContext ctx(false);
	Owned o;
	o.rv = (float*)malloc(sizeof(float) * 3 * nverts);
	o.areas = (unsigned char*)calloc(ntris, 1);
	if (!o.rv || !o.areas)
	{
		set_err("out of memory");
		return 0;
	}
	for (int i = 0; i < nverts; ++i)
	{
		o.rv[i * 3 + 0] = verts[i * 3 + 0];
		o.rv[i * 3 + 1] = verts[i * 3 + 2];
		o.rv[i * 3 + 2] = -verts[i * 3 + 1];
	}
	float bmin[3], bmax[3];
	rcCalcBounds(o.rv, nverts, bmin, bmax);
	int gw = 0, gh = 0;
	rcCalcGridSize(bmin, bmax, cs, &gw, &gh);

	o.hf = rcAllocHeightfield();
	if (!o.hf || !rcCreateHeightfield(&ctx, *o.hf, gw, gh, bmin, bmax, cs, ch))
	{
		set_err("rcCreateHeightfield failed");
		return 0;
	}
	rcMarkWalkableTriangles(&ctx, cfg[2], o.rv, nverts, tris, ntris, o.areas);
	if (!rcRasterizeTriangles(&ctx, o.rv, nverts, tris, o.areas, ntris, *o.hf, walkableClimb))
	{
		set_err("rcRasterizeTriangles failed");
		return 0;
	}
	rcFilterLowHangingWalkableObstacles(&ctx, walkableClimb, *o.hf);
	if (cfg[11] != 0.0f)
		rcFilterLedgeSpans(&ctx, walkableHeight, walkableClimb, *o.hf);
	rcFilterWalkableLowHeightSpans(&ctx, walkableHeight, *o.hf);

	o.chf = rcAllocCompactHeightfield();
	if (!o.chf || !rcBuildCompactHeightfield(&ctx, walkableHeight, walkableClimb, *o.hf, *o.chf))
	{
		set_err("rcBuildCompactHeightfield failed");
		return 0;
	}
	if (nbrushes > 0)
		carve_solids(*o.chf, planes, counts, nbrushes);
	if (!rcErodeWalkableArea(&ctx, walkableRadius, *o.chf))
	{
		set_err("rcErodeWalkableArea failed");
		return 0;
	}
	if (!rcBuildDistanceField(&ctx, *o.chf) ||
	    !rcBuildRegions(&ctx, *o.chf, 0, (int)cfg[8], (int)cfg[9]))
	{
		set_err("region build failed");
		return 0;
	}
	o.cset = rcAllocContourSet();
	if (!o.cset || !rcBuildContours(&ctx, *o.chf, cfg[7], (int)(cfg[6] / cs), *o.cset))
	{
		set_err("rcBuildContours failed");
		return 0;
	}
	o.pmesh = rcAllocPolyMesh();
	if (!o.pmesh || !rcBuildPolyMesh(&ctx, *o.cset, nvp, *o.pmesh))
	{
		set_err("rcBuildPolyMesh failed");
		return 0;
	}

	const rcPolyMesh& pm = *o.pmesh;
	rcw_result* r = (rcw_result*)calloc(1, sizeof(rcw_result));
	if (!r)
	{
		set_err("out of memory");
		return 0;
	}
	r->nverts = pm.nverts;
	r->npolys = pm.npolys;
	r->nvp = pm.nvp;
	r->grid_w = gw;
	r->grid_h = gh;
	r->verts = (float*)malloc(sizeof(float) * 3 * (pm.nverts > 0 ? pm.nverts : 1));
	r->polys = (int*)malloc(sizeof(int) * 2 * pm.nvp * (pm.npolys > 0 ? pm.npolys : 1));
	r->regions = (int*)malloc(sizeof(int) * (pm.npolys > 0 ? pm.npolys : 1));
	if (!r->verts || !r->polys || !r->regions)
	{
		rcw_free(r);
		set_err("out of memory");
		return 0;
	}
	for (int i = 0; i < pm.nverts; ++i)
	{
		const unsigned short* v = &pm.verts[i * 3];
		const float rx = pm.bmin[0] + v[0] * pm.cs;
		const float ry = pm.bmin[1] + v[1] * pm.ch;
		const float rz = pm.bmin[2] + v[2] * pm.cs;
		r->verts[i * 3 + 0] = rx;
		r->verts[i * 3 + 1] = -rz;
		r->verts[i * 3 + 2] = ry;
	}
	for (int i = 0; i < pm.npolys * 2 * pm.nvp; ++i)
		r->polys[i] = pm.polys[i] == RC_MESH_NULL_IDX ? -1 : (int)pm.polys[i];
	for (int i = 0; i < pm.npolys; ++i)
		r->regions[i] = pm.regs[i];
	*out = r;
	return 1;
}
