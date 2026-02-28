# ============================================================================
# Final Year Project: ML-Based Predictive Congestion Control Using NS2
# File: topology.tcl
# Description: NS2 simulation creating congestion scenarios for ML training
# ============================================================================

# Create a new simulator instance
set ns [new Simulator]

# Define colors for data flows (for visualization in NAM)
$ns color 1 Blue
$ns color 2 Red
$ns color 3 Green
$ns color 4 Yellow

# Open trace file for writing
set tracefile [open out.tr w]
$ns trace-all $tracefile

# ============================================================================
# TOPOLOGY CREATION: 6-node topology with bottleneck link
# ============================================================================

# Create 6 nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]
set n5 [$ns node]

# ============================================================================
# LINK CONFIGURATION
# ============================================================================

# Links from sources (n0, n1) to router n2 - High bandwidth
$ns duplex-link $n0 $n2 10Mb 5ms DropTail
$ns duplex-link $n1 $n2 10Mb 5ms DropTail

# BOTTLENECK LINK (n2 to n3) - Read capacity from argument
if { $argc == 1 } {
    set bw [lindex $argv 0]Mb
} else {
    set bw 2Mb
}

$ns duplex-link $n2 $n3 $bw 10ms DropTail
$ns queue-limit $n2 $n3 20

# Links from router n3 to destinations (n4, n5) - High bandwidth
$ns duplex-link $n3 $n4 10Mb 5ms DropTail
$ns duplex-link $n3 $n5 10Mb 5ms DropTail

# ============================================================================
# UDP AGENTS AND APPLICATIONS
# ============================================================================

# Get LOAD_TYPE from environment variables
if { [info exists env(LOAD_TYPE)] } {
    set load_type $env(LOAD_TYPE)
} else {
    set load_type "MEDIUM"
}

proc setup_cbr { ns node_src node_dst start_time stop_time class interval } {
    set udp [new Agent/UDP]
    $udp set class_ $class
    $ns attach-agent $node_src $udp
    
    set null [new Agent/Null]
    $ns attach-agent $node_dst $null
    $ns connect $udp $null
    
    set cbr [new Application/Traffic/CBR]
    $cbr attach-agent $udp
    $cbr set type_ CBR
    $cbr set packetSize_ 1000
    $cbr set interval_ $interval
    
    $ns at $start_time "$cbr start"
    $ns at $stop_time "$cbr stop"
}

# Determine interval dynamically based on LOAD_TYPE
if {$load_type == "LOW"} {
    set interval 0.02
} elseif {$load_type == "MEDIUM"} {
    set interval 0.01
} else {
    set interval 0.005
}

# Initialize the 4 traffic flows with varying start and stop times
setup_cbr $ns $n0 $n4 5.0 55.0 1 $interval
setup_cbr $ns $n1 $n5 5.0 55.0 2 $interval

# Burst traffic to guarantee periods of heavy congestion regardless of link capacity
# Interval 0.002 = 500 pkts/sec = 4 Mbps. Will congest even the 3 Mbps LOW load link.
setup_cbr $ns $n0 $n5 20.0 30.0 3 0.002
setup_cbr $ns $n1 $n4 40.0 50.0 4 0.002

# ============================================================================
# FINISH PROCEDURE
# ============================================================================

proc finish {} {
    global ns tracefile
    $ns flush-trace
    close $tracefile
    puts "Simulation completed successfully!"
    puts "Trace file generated: out.tr"
    exit 0
}

# Schedule simulation end
$ns at 60.0 "finish"

# ============================================================================
# RUN SIMULATION
# ============================================================================
puts "Starting NS2 simulation..."
puts "Load type: $load_type"
$ns run
