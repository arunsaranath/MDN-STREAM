# Control file for level 1A LAC data files
# Specifies what portions of the files should be verified
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
#  navigation discrepency (is line-to-line distance within bounds?)
#          low bound  high bound
#L1NAVDISC	1.0	1.25
#  ease up tolerences for nav yaw
#L1NAVDISC      .10     600.0
#  Try this to see what fails for Fred
L1NAVDISC      0.2     3.0 
#
#  tilt duration check (tilt change should last less than threshold)
#          threshold (sec)
 L1TILT		20.0
#
# End of file
