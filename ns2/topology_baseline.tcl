# ============================================================================
# BASELINE VERSION (NO ML)
# Files: baseline.tr / baseline.nam
# Total Time: 15 sec
# ============================================================================

set ns [new Simulator]
$ns rtproto DV
Agent/rtProto/DV set advertInterval 0.5

# --------------------------------------------------------------------------
# TRACE FILES
# --------------------------------------------------------------------------

set tracefile [open baseline.tr w]
$ns trace-all $tracefile

set namfile [open baseline.nam w]
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
# LINKS (Same as ML version)
# --------------------------------------------------------------------------

$ns duplex-link $n0 $n1 20Mb 5ms DropTail

$ns duplex-link $n1 $n2 2Mb 20ms DropTail
$ns queue-limit $n1 $n2 100

$ns duplex-link $n2 $n3 10Mb 5ms DropTail

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

# Initial colors (main path active)
$ns duplex-link-op $n1 $n2 color "red"
$ns duplex-link-op $n1 $n4 color "blue"
$ns duplex-link-op $n4 $n2 color "blue"

# --------------------------------------------------------------------------
# STATIC COSTS (Never Changed)
# --------------------------------------------------------------------------

$ns cost $n1 $n2 1
$ns cost $n2 $n1 1
$ns cost $n1 $n4 5
$ns cost $n4 $n1 5
$ns cost $n4 $n2 5
$ns cost $n2 $n4 5

# --------------------------------------------------------------------------
# TRAFFIC (Same as ML version)
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
# START EVENTS
# --------------------------------------------------------------------------

$ns at 0.5 "$cbr1 start"
$ns at 0.7 "$cbr2 start"

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
    exec nam baseline.nam &
    exit 0
}

$ns run