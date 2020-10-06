algorithm.py has been updated to contain two new filters for Soil Moisture.

By calibrating with calib_SM2RAIN the original behaviour of SM2RAIN algorithm will be mantained, by obtaining a value for the three parameters Z, a and b.
By calibrating with calib_SM2RAIN_T an exponential filter parameter T will be optimized to reduce Soil Moisture Noise, obtaining therefore four parameters Z, a, b and T.
By calibrating with calib_SM2RAIN_Tpot a modified exponential filter with two parameters, T and c, will be optimized to reduce Soil Moisture Noise according to Soil Moisture level, obtaining therefore five parameters Z, a, b, T and c.

ts_SM2RAIN will give as output the estimated rainfall, adapting to the number of parameters given as input
