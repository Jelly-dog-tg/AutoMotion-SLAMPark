"""Independent log-odds mapping of cached Lidar2D measurements."""
import json
import math
import numpy as np


class OccupancyGridMapper:
    """Rows index y, columns index x; states: unknown=-1, free=0, occupied=1."""

    def __init__(self, width, height, resolution=0.05, origin_x=0., origin_y=0.,
                 hit_probability=0.7, miss_probability=0.4,
                 probability_min=0.03, probability_max=0.97,
                 free_threshold=0.35, occupied_threshold=0.65):
        values = (width, height, resolution, origin_x, origin_y)
        if not all(math.isfinite(v) for v in values) or min(width, height, resolution) <= 0:
            raise ValueError('Invalid map geometry')
        if not 0 < probability_min < free_threshold < 0.5 < occupied_threshold < probability_max < 1:
            raise ValueError('Invalid probability thresholds')
        if not 0 < miss_probability < 0.5 < hit_probability < 1:
            raise ValueError('Invalid sensor probabilities')
        self.config = dict(width=width, height=height, resolution=resolution,
                           origin_x=origin_x, origin_y=origin_y,
                           hit_probability=hit_probability, miss_probability=miss_probability,
                           probability_min=probability_min, probability_max=probability_max,
                           free_threshold=free_threshold, occupied_threshold=occupied_threshold)
        self.width, self.height, self.resolution = width, height, resolution
        self.origin_x, self.origin_y = origin_x, origin_y
        self.rows, self.cols = math.ceil(height/resolution), math.ceil(width/resolution)
        self.log_odds = np.zeros((self.rows, self.cols), dtype=float)
        self.observed = np.zeros_like(self.log_odds, dtype=bool)

    def world_to_grid(self, x, y):
        return math.floor((x-self.origin_x)/self.resolution), math.floor((y-self.origin_y)/self.resolution)

    def grid_to_world(self, col, row):
        return self.origin_x+(col+0.5)*self.resolution, self.origin_y+(row+0.5)*self.resolution

    def contains(self, col, row):
        return 0 <= col < self.cols and 0 <= row < self.rows

    @staticmethod
    def ray_cells(start, end):
        """Bresenham traversal, including both endpoints."""
        x, y = start
        ex, ey = end
        dx, dy = abs(ex-x), -abs(ey-y)
        sx, sy = (1 if x < ex else -1), (1 if y < ey else -1)
        error = dx+dy
        while True:
            yield x, y
            if (x, y) == (ex, ey):
                break
            twice = 2*error
            if twice >= dy:
                error += dy
                x += sx
            if twice <= dx:
                error += dx
                y += sy

    def update(self, pose, scan, lidar_config):
        """False valid flags in Lidar2D mean max-range misses, not dropouts.

        Only explicit finite max-range misses clear space; malformed/ambiguous
        measurements are skipped. Each cell receives at most one update per scan,
        with occupied endpoints taking priority over free traversals.
        """
        free, occupied = set(), set()
        start = self.world_to_grid(pose.x, pose.y)
        for angle, distance, hit in zip(scan['angles_rad'], scan['ranges_m'], scan['valid'], strict=True):
            if not np.isfinite(angle) or not np.isfinite(distance):
                continue
            if not lidar_config.min_range_m <= distance <= lidar_config.max_range_m:
                continue
            if not hit and not np.isclose(distance, lidar_config.max_range_m, rtol=0, atol=1e-9):
                continue
            theta = pose.yaw+angle
            end = self.world_to_grid(pose.x+distance*math.cos(theta), pose.y+distance*math.sin(theta))
            cells = list(self.ray_cells(start, end))
            free.update(cell for cell in (cells[:-1] if hit else cells) if self.contains(*cell))
            if hit and self.contains(*end):
                occupied.add(end)
        free.difference_update(occupied)
        for cells, probability in ((free, self.config['miss_probability']), (occupied, self.config['hit_probability'])):
            if cells:
                cols, rows = np.array(list(cells)).T
                self.log_odds[rows, cols] += math.log(probability/(1-probability))
                self.observed[rows, cols] = True
        lo, hi = self.config['probability_min'], self.config['probability_max']
        np.clip(self.log_odds, math.log(lo/(1-lo)), math.log(hi/(1-hi)), out=self.log_odds)

    @property
    def probabilities(self):
        return 1/(1+np.exp(-self.log_odds))

    @property
    def states(self):
        p = self.probabilities
        states = np.full(p.shape, -1, dtype=np.int8)
        states[self.observed & (p <= self.config['free_threshold'])] = 0
        states[self.observed & (p >= self.config['occupied_threshold'])] = 1
        return states

    def arrays(self, prefix=''):
        return {prefix+'config': np.array(json.dumps(self.config)),
                prefix+'log_odds': self.log_odds, prefix+'observed': self.observed,
                prefix+'states': self.states, prefix+'probabilities': self.probabilities}

    def save(self, path):
        np.savez_compressed(path, **self.arrays())

    @classmethod
    def load(cls, path, prefix=''):
        with np.load(path, allow_pickle=False) as data:
            mapper = cls(**json.loads(str(data[prefix+'config'])))
            mapper.log_odds[:] = data[prefix+'log_odds']
            mapper.observed[:] = data[prefix+'observed']
        return mapper
