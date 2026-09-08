# Control file for level 1A GAC data files
# Specifies what portions of the files should be verified
#  30 oct 2003 set all the TDI and GAIN error threhold % to .2
#
# % saturated pixels for gain 1
#              error thresh  (%)
L1GAIN11  	50.0
L1GAIN12  	50.0
L1GAIN13  	50.0
L1GAIN14  	50.0
L1GAIN15  	50.0
L1GAIN16  	50.0
L1GAIN17  	50.0
L1GAIN18  	50.0
#
# % saturated pixels for gain 2
#             error thresh (%)
#
L1GAIN21  	50.0
L1GAIN22  	50.0
L1GAIN23  	50.0
L1GAIN24  	50.0
L1GAIN25  	50.0
L1GAIN26  	50.0
L1GAIN27  	50.0
L1GAIN28  	50.0
#
# Zero pixel %
#           error thresh (%)
L1ZERO1		50.0
L1ZERO2		50.0
L1ZERO3		50.0
L1ZERO4		50.0
L1ZERO5		50.0
L1ZERO6		50.0
L1ZERO7		50.0
L1ZERO8		50.0
#
# % of pixels with count above threshold
#          err thresh (%)  threshold (counts)
L1HICOUNT1	1.0	1024.0
L1HICOUNT2	1.0	1024.0
L1HICOUNT3	1.0	1024.0
L1HICOUNT4	1.0	1024.0
L1HICOUNT5	1.0	1024.0
L1HICOUNT6	1.0	1024.0
L1HICOUNT7	1.0	1024.0
L1HICOUNT8	1.0	1024.0
#
#  % of pixels with count below threshold
#         err thresh (%)  threshold (counts)
L1LOWCOUNT1	1.0  	0.0
L1LOWCOUNT2	1.0	0.0
L1LOWCOUNT3	1.0	0.0
L1LOWCOUNT4	1.0	0.0
L1LOWCOUNT5	1.0	0.0
L1LOWCOUNT6	1.0	0.0
L1LOWCOUNT7	1.0	0.0
L1LOWCOUNT8	1.0	0.0
#
# the TLIM_NEGTIM is a time-dependent fail control.  anything older than
# the time (in YYYYDDDHHMMSS format) will not get a fail status
#  current value to start of time shift recording (Fall, 2001)
TLIM_NEGTIM 2001333220000
#
#  navigation discrepency (is line-to-line distance within bounds?)
#  this is now in milliradians / sec for orbit of 98.9 min, should \
#  be 1.059 , modified to account for earth spin = 1.07, so try +- 
#  10%
# old: L1NAVDISC       0.01 150.0
L1NAVDISC       0.97 1.17
#
#  time range check of the file to see if it is in a check range
TRNG_CHK
#
#  tilt duration check (tilt change should last less than threshold)
#          threshold (sec)
 L1TILT		20.0
#
#  noise % limits for reporting regular and encryption noise
NOISE      80
ENCRYPT     70
#
# Time limit for the instrument analog check
#   all other TLIM times set to R4 start
TLIM_INST_ANA 2002177000000
#
# new as of 31 oct 2001, for instrument analog telemetry, the low, high 
# thresholds and the % acceptable outside these thresholds
#  Current values derived from a sample of GAC from mission start through 2000
#  usually normal min, max +- 25% 
#  26 Aug 09  up values on following: 06: hi from 15.75 to 25, 09: low 
#  from 1.975 to -1.8, 14: hi from 45 to 55, 15: hi from 40 to 50, 
#  16: hi from 40 to 50
#  11 Nov 2009 some tuning of limits and generally exchange ranges for 'A' 
#  and 'B' items (9&10, 25&26, 27&28, 29&30, 31&32) due to use of side B 
#  electronics instead of side A
INST_ANA01  5.0  25.0  1.   focal plane t 1
INST_ANA02  5.0  25.0  1.   focal plane t 2
INST_ANA03  5.0  25.0  1.   focal plane t 3
INST_ANA04  5.0  25.0  1.   focal plane t 4
INST_ANA05  5.0  25.0  1.   telescope motor t
INST_ANA06  -0.75  25.00  1.  tilt platform t
INST_ANA07  1.0  20.438  1.  tilt platform t
INST_ANA08  5.  25.  1.  half angle motor t
INST_ANA09  0.41  0.53  1.  power supply in current B
INST_ANA10  1.800  3.405  1.  power supply in current A
INST_ANA11  13.72  16.73  1.  analog power voltage +15 V
INST_ANA12  -16.95  -13.95  1.  analog power voltage -15 V
INST_ANA13  5.09  5.26  1.  5 V logical power voltage
INST_ANA14  33.35  55.00  1.  power supply t
INST_ANA15  14.87  50.00  1.  B1/B2 postamp t
INST_ANA16  25.0  50.0  1.  servo drive t
INST_ANA17  28.0   32.20  1.  servo power voltage: +30 V
INST_ANA18  20.03  20.72  1.  servo power voltage: +21 V
INST_ANA19  -20.78  -20.0  1.  servo power voltage: -21 V
INST_ANA20  4.79  5.86  1.  servo power voltage: +5 V
INST_ANA21  1183.  1255.  1.  angular momentum compensator speed
INST_ANA22  47.  204.  1.  tilt motor 2 position, platform
INST_ANA23  46.00  244.00  1.  tilt motor 2 position, base
INST_ANA24  22.00  34.00  1.  28 V heater power
INST_ANA25  0.018  0.035 1.  telescope motor A current
INST_ANA26  0.067 0.26 1.  telescope motor B current
INST_ANA27  0.017  0.038  1.  half angle motor A current
INST_ANA28  0.060 0.172 1.  half angle motor B current
INST_ANA29  -1.18  -1.09  1.  servo phase error A
INST_ANA30  -0.13 -0.02  1.  servo phase error B
INST_ANA31  0.125  0.225  1.  angular momentum compensator current A
INST_ANA32  0.30  0.512  1.  angular momentum compensator current B
#
# Time limit for the gain value check
TLIM_GAINV_CHK 2002177000000
#
# gain value check and accptable thresholds
GAINV_CHK1  0.20     band 1 gain check and acceptable % not at specified value
GAINV_CHK2  0.20     band 2
GAINV_CHK3  0.20     band 3
GAINV_CHK4  0.20     band 4
GAINV_CHK5  0.20     band 5
GAINV_CHK6  0.20     band 6
GAINV_CHK7  0.20     band 7
GAINV_CHK8  0.20     band 8
#
# Time limit for the tdi value check
TLIM_TDIV_CHK 2002177000000
#
# tdi value check and accptable thresholds
TDIV_CHK1  0.20     band 1 tdi check and acceptable % not at specified value
TDIV_CHK2  0.20     band 2
TDIV_CHK3  0.20     band 3
TDIV_CHK4  0.20     band 4
TDIV_CHK5  0.20     band 5
TDIV_CHK6  0.20     band 6
TDIV_CHK7  0.20     band 7
TDIV_CHK8  0.20     band 8
#
# End of file
