# ============================================================================
# LOGICALLY CLEAN ML CONGESTION CONTROL (HORIZONTAL LAYOUT)
# Total Time: 15 sec
# ============================================================================

set ns [new Simulator]
$ns rtproto DV
Agent/rtProto/DV set advertInterval 0.5

# --------------------------------------------------------------------------
# TRACE FILES
# --------------------------------------------------------------------------

set tracefile [open ml.tr w]
$ns trace-all $tracefile

set namfile [open ml.nam w]
$ns namtrace-all $namfile

$ns color 1 Blue
$ns color 2 Red

# --------------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------------

set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]

# --------------------------------------------------------------------------
# LINKS
# --------------------------------------------------------------------------

# Strong source link
$ns duplex-link $n0 $n1 20Mb 5ms DropTail

# Moderate bottleneck
$ns duplex-link $n1 $n2 2Mb 20ms DropTail
$ns queue-limit $n1 $n2 100

# Destination
$ns duplex-link $n2 $n3 10Mb 5ms DropTail

# Strong alternate
$ns duplex-link $n1 $n4 15Mb 10ms DropTail
$ns duplex-link $n4 $n2 15Mb 10ms DropTail
$ns queue-limit $n1 $n4 100
$ns queue-limit $n4 $n2 100

# --------------------------------------------------------------------------
# FORCE HORIZONTAL LAYOUT
# --------------------------------------------------------------------------

$ns duplex-link-op $n0 $n1 orient right
$ns duplex-link-op $n1 $n2 orient right
$ns duplex-link-op $n2 $n3 orient right
$ns duplex-link-op $n1 $n4 orient down
$ns duplex-link-op $n4 $n2 orient right-up

# Initial colors
$ns duplex-link-op $n1 $n2 color "red"
$ns duplex-link-op $n1 $n4 color "blue"
$ns duplex-link-op $n4 $n2 color "blue"

# --------------------------------------------------------------------------
# COSTS
# --------------------------------------------------------------------------

$ns cost $n1 $n2 1
$ns cost $n2 $n1 1
$ns cost $n1 $n4 5
$ns cost $n4 $n1 5
$ns cost $n4 $n2 5
$ns cost $n2 $n4 5

# --------------------------------------------------------------------------
# TRAFFIC
# --------------------------------------------------------------------------

set udp1 [new Agent/UDP]
$udp1 set class_ 1
$ns attach-agent $n0 $udp1

set null1 [new Agent/Null]
$ns attach-agent $n3 $null1
$ns connect $udp1 $null1

set cbr1 [new Application/Traffic/CBR]
$cbr1 attach-agent $udp1
$cbr1 set packetSize_ 1000
$cbr1 set interval_ 0.005

set udp2 [new Agent/UDP]
$udp2 set class_ 2
$ns attach-agent $n0 $udp2

set null2 [new Agent/Null]
$ns attach-agent $n3 $null2
$ns connect $udp2 $null2

set cbr2 [new Application/Traffic/CBR]
$cbr2 attach-agent $udp2
$cbr2 set packetSize_ 1000
$cbr2 set interval_ 0.005

# --------------------------------------------------------------------------
# ML FLAG
# --------------------------------------------------------------------------

set ml_triggered 0

proc ml_check {} {
    global ns n1 n2 n4 ml_triggered cbr1 cbr2

    set now [$ns now]
    $ns flush-trace

    set prob [exec python3 ../scripts/predict.py ml.tr $now]
    puts "Time $now : Prob = $prob"

    if {$prob > 0.4 && $ml_triggered == 0} {

        puts "ML TRIGGERED → REROUTE"

        $ns cost $n1 $n2 50
        $ns cost $n2 $n1 50

        $ns duplex-link-op $n1 $n2 color "gray"
        $ns duplex-link-op $n1 $n4 color "green"
        $ns duplex-link-op $n4 $n2 color "green"

        # slight stabilization
        $cbr1 set interval_ 0.006
        $cbr2 set interval_ 0.006

        set ml_triggered 1
    }

    $ns at [expr {$now + 0.2}] "ml_check"
}

# --------------------------------------------------------------------------
# START EVENTS
# --------------------------------------------------------------------------

$ns at 0.5 "$cbr1 start"
$ns at 0.7 "$cbr2 start"

# Delay ML start (important fix)
$ns at 2.0 "ml_check"

# Increase load gradually
$ns at 5.0 {
    $cbr1 set interval_ 0.003
    $cbr2 set interval_ 0.003
}

$ns at 12.0 {
    $cbr1 stop
    $cbr2 stop
}

$ns at 15.0 {
    $ns flush-trace
    close $tracefile
    close $namfile
    exec nam ml.nam &
    exit 0
}

$ns run