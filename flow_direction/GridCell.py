import os
from rasterio import mask
import shapely
import numpy as np

class GridCell:
    def __init__(self, geom, rast):
        """This class represents one hydrological grid-cell. When initialized with a polygon 
        geometry and a raster file, this class estimates the Grid cell direction based on the 
        algorithm developed by Donnel et. al., 1999.
        
        PARAMETERS:
            geom   : a geopandas object, single polygon which can be used to clip rasters
            rast   : a rasterio raster object
        """
        self._geom = geom["geometry"]       # Extract the geometry
        self._id = geom.name
        self._cellsize = 0.0625             # Hardcoded cellsize in georeferenced units
        
        self.src = rast
        
        self.direction_dict = {             # Helper dictionary, can be used to convert 
            1: 'N',                         # numbers to letter directions
            2: 'NE',
            3: 'E',
            4: 'SE',
            5: 'S',
            6: 'SW',
            7: 'W',
            8: 'NW'
        }
        self.direction_dict_r = {           # Helper dictionary, can be used to convert
            'N': 1,                         # letter directions to numbers
            'NE': 2,
            'E': 3,
            'SE': 4,
            'S': 5,
            'SW': 6,
            'W': 7,
            'NW': 8
        }
        
        self.determine_direction()          # This calculates the directions, and initiates
                                            # a self.griddirection 
        
    def determine_direction(self):
        """Determines the directions based on the algorithm proposed by Donnel et. al., 1999
        """
        # Clip the raster according to geometry
        self.clipped = self.clip()

        # Determine the zones of gridcell, which define how our directions are determined
        self._leftlim = int(self.clipped.shape[1] * 1/3)
        self._rightlim = int(self.clipped.shape[1] * 2/3)
        self._toplim = int(self.clipped.shape[0] * 1/3)
        self._bottomlim = int(self.clipped.shape[0] * 2/3)
        
        # Edge indices
        self.edges = (0, self.clipped.shape[0]-1, self.clipped.shape[1]-1)

        # Determine the index of cell from which the flow exists
        self.exit = self.exit_pt()
        
        # Determine the direction using naive direction function
        self.griddirection = self.naive_direction()
        
        # If the exit pixel lies in any of the corners, perform additional steps
        if self.griddirection in (2, 4, 6, 8):
            # Get geometry of the neighbouring region
            neighbour_geom = self.get_neighbour(self.griddirection)
            
            # Clip the raster using neighbouring geometry
            clipped = self.clip(neighbour_geom)
            
            # Determine the exit indices in the neighbouring geometry
            exit_x, exit_y = self.exit_pt(clipped)
            
            # Determine the direction of exit in the neighbouring region
            neighbour_d = None
            
            if exit_y <= clipped.shape[1]/2:
                neighbour_d = 'N'
            else:
                neighbour_d = 'S'
            
            if exit_x <= clipped.shape[0]/2:
                neighbour_d = neighbour_d+'W'
            else:
                neighbour_d = neighbour_d+'E'
            
            # Convert determined direction from alphabets to number
            current_d = self.direction_dict[self.griddirection]
            
            # Based on how the stream flows in the neighbouring region, set the appropriate 
            # direction of the current grid-cell. The function self.lookdeeper(...) is used for 
            # determining the direction in this case.
            self.griddirection = self.direction_dict_r[self.lookdeeper(current_d, neighbour_d)]
        
        return self.griddirection


    def clip(self, geom=None):
        """Helper function that clips the base raster for any given geometry

        PARAMETERS:
            geom:   optionally, a polygon geometry can be passed which'll be used to clip
        """
        # If geometry is not passed, use the gridcell's geometry
        if geom is None:
            geom = self._geom
        masked_band, masked_transform = mask.mask(self.src, [geom], crop=True)

        return masked_band[0]
    
    def exit_pt(self, clipped=None):
        """Return the index of pixel from which the stream exits the grid cell
        """
        if clipped is None:
            clipped = self.clipped
        wh = np.where(clipped == clipped.max())
        return wh[0][0], wh[1][0]
    
    def clean_exit(self):
        """Return if the exit is "clean" - if the highest flow accumulation point is at the edge
            of the clipped raster. If the highest accumulation point is anywhere other than the
            edge, something is wrong at that grid cell - most likely it overlies the River itself
        """
        return (self.exit[0] in self.edges) or (self.exit[1] in self.edges)
    
    def naive_direction(self):
        """Determines the direction naively, i.e., solely based on the index of the pour point of
            the stream. This doesn't take into consideration the neigbourhood of the exit point
        """
        directions = {
            'N': 1,
            'NE': 2,
            'E': 3,
            'SE': 4,
            'S': 5,
            'SW': 6,
            'W': 7,
            'NW': 8,
            'dirty': 0
        }
        if not self.clean_exit():
            return directions['dirty']
        else:
            y, x = self.exit
            
            if x < self._leftlim:
                if y < self._toplim:
                    return directions['NW']
                elif y < self._bottomlim:
                    return directions['W']
                else:
                    return directions['SW']
            elif x < self._rightlim:
                if y < self._toplim:
                    return directions['N']
                else:
                    return directions['S']
            else:
                if y < self._toplim:
                    return directions['NE']
                elif y < self._bottomlim:
                    return directions['E']
                else:
                    return directions['SE']
                
    def get_neighbour(self, direction):
        """For any of the non cardinal directions, get the geometry that defines the neighbourhood
            in that direction. So for NE direction, the region would be a polygon that is offset 
            on both X and Y directions by half the grid cell size.
        """
        if direction == 2:        # NE
            return shapely.affinity.translate(self._geom, xoff=self._cellsize/2, yoff=self._cellsize/2)
        elif direction == 4:      # SE
            return shapely.affinity.translate(self._geom, xoff=self._cellsize/2, yoff=-self._cellsize/2)
        elif direction == 6:      # SW
            return shapely.affinity.translate(self._geom, xoff=-self._cellsize/2, yoff=-self._cellsize/2)
        elif direction == 8:      # NW
            return shapely.affinity.translate(self._geom, xoff=-self._cellsize/2, yoff=self._cellsize/2)
        
    def lookdeeper(self, current, neighbour):
        """This function performs additional if-else checks on the neighbourhood region to 
            determine if the stream indeed continues in the non-cardinal direction or if it 
            meanders to any other direction.
        """
        result = None
        if current == 'NE':
            if neighbour == 'NW':
                result = 'N'
            elif neighbour == 'SE':
                result = 'E'
            else:
                result = current
        elif current == 'SE':
            if neighbour == 'NE':
                result = 'E'
            elif neighbour == 'SW':
                result = 'S'
            else:
                result = current
        elif current == 'SW':
            if neighbour == 'NW':
                result = 'W'
            elif neighbour == 'SE':
                result = 'S'
            else:
                result = current
        elif current == 'NW':
            if neighbour == 'NE':
                result = 'N'
            elif neighbour == 'SW':
                result = 'W'
            else:
                result = current
        return result
