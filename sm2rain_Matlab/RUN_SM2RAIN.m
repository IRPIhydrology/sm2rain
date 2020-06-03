% ... few lines to run SM2RAIN
% Application with Cerbara dataset, 24-hour rainfall estimation from hourly observations

name='CER_1hour_2011';                          % name of the input file
AGGR=24;                                        % aggregation period: 24 hour=1 day
cal_SM2RAIN(name,AGGR)                          % SM2RAIN calibration
SM2RAIN(name,load(['PAR_',name,'.dat']),AGGR,1) % RUN SM2RAIN