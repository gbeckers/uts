
import time
import hashlib

def create_licensefile(filename, startyear=None, startmonth=None, endyear=None,
    endmonth=None, password='neostriatum'):

    with open(filename, 'w') as f:
        for yearmonth in _generate_months(startyear=startyear,
                                          startmonth=startmonth,
                                          endyear=endyear, endmonth=endmonth):
            datestr = '%s%s'% (yearmonth,password)
            print "adding license %s" % datestr
            f.write(hashlib.sha256(datestr).hexdigest()+'\n')

def check_licensefile(filename, startyear=None, startmonth=None, endyear=None,
                      endmonth=None, password='neostriatum'):

    lines = []
    with open(filename, 'r') as f:
         lines = [line.rstrip() for line in f.readlines()]
    for yearmonth in _generate_months(startyear=startyear,
                                          startmonth=startmonth,
                                          endyear=endyear, endmonth=endmonth):
        datestr = '%s%s'% (yearmonth,password)
        # we check for every line if it matches the teststr
        for line in lines:
            if hashlib.sha256(datestr).hexdigest() == line:
                print datestr, 'hit!'

def _generate_months(startyear=None, startmonth=None, endyear=None,
                      endmonth=None):

    ts = time.localtime()
    year = ts.tm_year
    month = ts.tm_mon
    if startyear is None:
        startyear = year
    if endyear is None:
        endyear = year
    if startmonth is None:
        startmonth = month
    if endmonth is None:
        endmonth = month

    for year in range(startyear, endyear+1):
        if (startyear != endyear) and (year != endyear):
            tomonth = 12
        else:
            tomonth = endmonth
        if (startyear != endyear) and (year != startyear):
            frommonth = 1
        else:
            frommonth = startmonth
        for month in range(frommonth,tomonth+1):
            yield '%d%.2d' % (year, month)

