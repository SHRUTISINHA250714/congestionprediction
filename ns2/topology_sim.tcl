# ============================================================================
# NS2 Topology - SIMULATION GENERATOR
# ============================================================================

set ns [new Simulator]
$ns rtproto DV

if { $argc != 4 } {
    puts "Usage: ns topology_sim.tcl <bottleneck_bw(Mb)> <queue_size> <tcp_window> <app_rate(Mb)>"
    exit 1
}

set bw [lindex $argv 0]Mb
set qsize [lindex $argv 1]
set tcp_win [lindex $argv 2]
set app_rate [lindex $argv 3]Mb

set tf [open out.tr w]
$ns trace-all $tf

proc finish {} {
    global ns tf
    $ns flush-trace
    close $tf
    exit 0
}

# Nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]
set n4 [$ns node]

# Links
$ns duplex-link $n0 $n1 10Mb 5ms DropTail
$ns duplex-link $n1 $n2 $bw 20ms DropTail
$ns queue-limit $n1 $n2 $qsize
$ns duplex-link $n2 $n3 10Mb 5ms DropTail

$ns duplex-link $n1 $n4 2Mb 10ms DropTail
$ns duplex-link $n4 $n2 2Mb 10ms DropTail
$ns queue-limit $n1 $n4 20
$ns queue-limit $n4 $n2 20

# App 1
set tcp1 [new Agent/TCP]
$tcp1 set window_ $tcp_win
$ns attach-agent $n0 $tcp1
set sink1 [new Agent/TCPSink]
$ns attach-agent $n3 $sink1
$ns connect $tcp1 $sink1

set app1 [new Application/Traffic/CBR]
$app1 attach-agent $tcp1
$app1 set rate_ $app_rate

# App 2 - Bursty background traffic
set tcp2 [new Agent/TCP]
$tcp2 set window_ [expr $tcp_win * 2]
$ns attach-agent $n0 $tcp2
set sink2 [new Agent/TCPSink]
$ns attach-agent $n3 $sink2
$ns connect $tcp2 $sink2

set app2 [new Application/Traffic/CBR]
$app2 attach-agent $tcp2
set burst_rate [expr [lindex $argv 3] * 3.0]Mb
$app2 set rate_ $burst_rate

# Schedule events to guarantee MIXED BEHAVIOR
# 0 - 10s: Low traffic (only App1 starts later)
$ns at 5.0 "$app1 start"

# 25 - 35s: Heavy burst (App2 joins with 3x rate)
$ns at 25.0 "$app2 start"
$ns at 35.0 "$app2 stop"

# 55 - 60s: No traffic (guaranteed 0 utilization)
$ns at 55.0 "$app1 stop"
$ns at 60.0 "finish"

$ns run
