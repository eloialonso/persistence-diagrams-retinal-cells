class Point:
    def __init__(self, x=0, y=0, cc=None):
        self.x = x
        self.y = y
        if cc is None:
            self.belongs_to = []
        else:
            self.belongs_to = cc

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)
