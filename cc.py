#! /usr/bin/env python
# coding: utf-8

"""
TODO
"""

class Point:
    def __init__(self, x=0, y=0, cc=None):
        """2D-Point that belongs to a particular connected component
        Arguments:
            x (int): x-position
            y (int): y-position
            cc (ConnectedComponent): cc the Point belongs to
        """
        self.x = x
        self.y = y
        if cc is None:
            self.belongs_to = []
        else:
            self.belongs_to = cc

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)


class ConnectedComponent: # TODO

    # Class attributes
    connected_components = []   # to store all the created instances
    history = {}                # to store TODO

    def __init__(self, peak=Point(), x_left=0, x_right=0, members=[]):
        r"""Connected component
        Arguments:
            peak (Point): highest point
            x_left (int): x-position of the left point of the cc
            x_right (int): x-position of the right point of the cc
            members (list): list of Point belonging to the cc
        """
        self.peak = peak
        self.x_left = x_left
        self.x_right = x_right
        self.is_inversed = x_left > x_right
        self.members = members
        if members == []:
            self.members.append(self.peak)

        # Update the two class attributes
        self.connected_components.append(self)  # Add this new cc to the list of cc
        self.history[self] = [self.peak.y]      # Register birth intensity

    def __repr__(self):
        inv = " (inversed) " if self.is_inversed else " "
        return "CC{}: peak=({}), x_left={}, x_right={}".format(inv, self.peak, self.x_left, self.x_right)

    @classmethod
    def reset(cls):
        r"""Remove all created cc and clear history"""
        for cc in cls.connected_components:
            del cc
        cls.connected_components = []
        cls.history = {}
    
    def add_member(self, point, point_close_left=False, point_close_right=False, on_left=False, on_right=False):
        r"""
        TODO
        """
        # add the new point to the list of members of the cc
        self.members.append(point)
        self.members = list(set(self.members))

        # if the new point is higher than the peak of the cc, it becomes the peak
        if self.peak.y < point.y:
            self.peak = point

        if self.is_inversed:
            if on_left:
                self.x_left = min(self.x_left, point.x)
            elif on_right:
                self.x_right = max(self.x_right, point.x)
            else:
                raise RuntimeError("This case is not expected...")
        else:
            if point_close_left:
                self.x_right = point.x
                self.is_inversed = True
            elif point_close_right:
                self.x_left = point.x
                self.is_inversed = True
            else:
                self.x_left = min(self.x_left, point.x)
                self.x_right = max(self.x_right, point.x)

    def is_close_to_point(self, point, resolution):
        on_right = False
        on_left = False
        is_close = False
        if self.is_inversed:
            if (point.x - resolution <= self.x_right):
                on_right = True
                is_close = True
            if (point.x + resolution >= self.x_left):
                on_left = True
                is_close = True
        else:
            is_close = (point.x + resolution >= self.x_left) and (point.x - resolution <= self.x_right)

        return (is_close, on_left, on_right)

    @staticmethod
    def union(cc1, cc2, intensity, stride):
        r"""Union of two connected components
        Arguments:
            cc1 and cc2:  the two cc we want to merge
            intensity: to store the death intensity of the lower cc
            stride: stride of the angular smoothing
        """
        
        if cc1 == cc2:
            return cc1
        
        # Identify which cc is higher than the other
        (cc_high, cc_low) = (cc1, cc2) if cc1.peak.y >= cc2.peak.y else (cc2, cc1)
        
        case1 = False
        case2 = False
        if cc2.x_left - cc1.x_right <= 2 * stride:
            case1 = cc1.is_inversed
            case2 = cc2.is_inversed
        elif cc1.x_left - cc2.x_right <= 2 * stride:
            case1 = cc2.is_inversed
            case2 = cc1.is_inversed
        if (cc1.is_inversed or cc2.is_inversed) and not (case1 or case2):
            raise RuntimeError("Not expected: union of inversed peak without case1 and case2")
        
        # Merge the lower cc in the higher cc
        cc_high.members = list(set(cc_high.members + cc_low.members))
        if case1:
            if cc_high.is_inversed:
                cc_high.x_right = max(cc_high.x_right, cc_low.x_right)
            elif cc_low.is_inversed:
                cc_high.is_inversed = True
                cc_high.x_left = cc_low.x_left
            else:
                raise RuntimeError("Not expected")
        elif case2:
            if cc_high.is_inversed:
                cc_high.x_left = min(cc_high.x_left, cc_low.x_left)
            elif cc_low.is_inversed:
                cc_high.is_inversed = True
                cc_high.x_right = cc_low.x_right
            else:
                raise RuntimeError("Not expected")
        else:
            cc_high.x_left = min(cc_high.x_left, cc_low.x_left)
            cc_high.x_right = max(cc_high.x_right, cc_low.x_right)
        
        # Remove cc_low from the list of cc
        ConnectedComponent.connected_components = [x for x in ConnectedComponent.connected_components if x != cc_low]
        
        # Register the death intensity of cc_low
        ConnectedComponent.history[cc_low].append(intensity)
            
        return cc_high


    
