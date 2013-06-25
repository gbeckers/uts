
import time
import hashlib

def create_licensefile(filename, lastyear=2012, lastmonth=8, startyear=None, 
    startmonth=None, password='neostriatum'):

    f = open(filename,'w')
    ts = time.localtime()
    if startyear is None:
        startyear = ts.tm_year
    if startmonth is None:
        startmonth = ts.tm_mon
    for year in range(startyear, lastyear+1):
        if (startyear != lastyear) and (year != lastyear):
            endmonth = 12
        else:
            endmonth = lastmonth
        if (startyear != lastyear) and (year != startyear):
            beginmonth = 1
        else:
            beginmonth = startmonth
        for month in range(beginmonth,endmonth+1):
            datestr = ('%d' % year) + ('%.2d' % month)+password
            print "adding license %s" % datestr
            f.write(hashlib.sha256(datestr).hexdigest()+'\n')
    f.close()

