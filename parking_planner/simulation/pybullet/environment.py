"""P11.3.1 peripheral scenery: visual only, outside the 22 m test domain."""


def build_background(scene):
    ids=[]
    def box(center,size,color):
        body=scene.box(center,size,color,collision=False)
        ids.append(body)
        return body
    # Same asphalt color. Surface is 5 mm below the original physical slab,
    # retaining its shape, height, friction and contact solver inputs exactly.
    box((11.,11.,-.105),(2000.,2000.,.2),(.20,.23,.26,1))
    paint=(.80,.83,.81,1)
    def line(start,end):
        body=scene.line(start,end,paint,.07)
        ids.append(body)
    def parked(x,y,color):
        box((x,y,.65),(1.8,4.5,1.3),color)
        box((x,y,1.47),(1.42,2.3,.34),(.12,.19,.23,1))
        for dx in (-.98,.98):
            line((x+dx,y-2.5),(x+dx,y+2.5))
        line((x-.98,y+2.5),(x+.98,y+2.5))
    for x,color in [(3.,(.50,.53,.55,1)),(10.,(.43,.29,.20,1)),(17.,(.24,.37,.43,1))]:
        parked(x,26.,color)
    for y,color in [(5.,(.48,.22,.23,1)),(15.,(.36,.41,.34,1))]:
        parked(-5.,y,color)
    for y in (5.,11.,17.):
        box((25.,y,.45),(.55,1.6,.9),(.58,.59,.56,1))
        box((25.,y,.92),(.56,1.62,.04),(.83,.67,.19,1))
    # Low sparse edge rails, separated from the experimental boundary.
    for x in (-8.,27.):
        for y in (0.,8.,16.,24.):
            box((x,y,.45),(.12,.12,.9),(.48,.50,.49,1))
        box((x,12.,.78),(.10,24.,.12),(.58,.60,.60,1))
    return ids
