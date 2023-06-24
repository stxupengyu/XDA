import time

def time_since(start_time):
    #adaptive return time cost
    cost = time.time() - start_time
    if cost<60:
        cost_str = '%.1f s'%cost
    elif cost<3600:
        cost_str = '%.1f mins' % (cost/60.0)
    else:
        cost_str = '%.1f hours' % (cost/3600.0)
    return cost_str

def cprint(name, value):
    print('=============%s=============='%name)
    print(value)
    print('=============%s=============='%name)