# Power and thermal on passively-cooled P100s

Not part of the patches, but it moved throughput more than patch 0001 did, and it
is the kind of thing that silently corrupts benchmarks. Measured on two
P100-PCIE-16GB in a Dell rack chassis.

## The P100 is thermally limited, not power limited

The cards are 250 W parts. They had been capped to 175 W to be safe on the mains
supply. Raising the cap to 250 W looked like free performance and was not.

Identical run (`llama-bench -p 0 -n 512 -r 3`), both from a matched 62 °C start:

| cap | tok/s | peak temp | `sw_thermal_slowdown` | steady clock | power |
|---|---|---|---|---|---|
| 250 W | 37.85 ± 2.25 | **79 °C** | **Active by 30 s** | falls to **1113 MHz** | 237 → 156 W |
| 175 W | 36.21 ± 0.51 | 76 °C | never engages | holds **1164-1202 MHz** | ~165 W |

At 250 W the card reaches 79 °C in about 30 seconds, `sw_thermal_slowdown` engages,
and the clock walks down *below* where the 175 W cap sustains it. The +4.5% is
entirely the first ~25 seconds before heat soak. On a heat-soaked card, back-to-back
runs measured **24.17** against 36.05 at 175 W — a 33% regression.

Peak system power at 250 W was only 627 W against a previous high of 655 W. **Power
was never the constraint.**

## The cause was the chassis fan policy

    racadm get system.thermalsettings
    ThermalProfile=Sound Cap        <- noise-limited
    FanSpeedOffset=Off

Under `Sound Cap` the fans reached 68% PWM (~12,960 RPM) under load while the GPU
throttled, with inlet at 33 °C and exhaust at 38 °C — the chassis had cooling
headroom it was deliberately not using.

Setting `ThermalProfile=Maximum Performance` and `FanSpeedOffset=Max` puts all six
fans at 100% (~17,700 RPM). This is **loud**. It is the whole trade.

With turbo fans, same test:

| cap | fans | tok/s | equilibrium | slowdown | clock |
|---|---|---|---|---|---|
| 175 W | Sound Cap | 36.21 | 76 °C | never | 1164-1202 MHz |
| 250 W | Sound Cap | — | 79 °C @ 30 s | **yes → 1113 MHz** | 1113 MHz |
| 175 W | turbo | 36.12 ± 0.51 | 74 °C stable | never | 1189-1202 MHz |
| **250 W** | **turbo** | **37.44 ± 1.33** | 79 °C setpoint | yes, floors higher | **1215-1227 MHz** |

Short runs gain much more: 250 W + turbo gave **39.24 ± 0.00** on a 512-token run
with the clock **pinned at 1328 MHz — max boost — for the whole run**, the only
configuration tested that never left maximum clock.

Idle temperature dropped from 63-74 °C to 49-52 °C.

**250 W with quiet fans is the one combination worse than either.** If you make the
box quiet again, put the cap back in the same change.

## 79 °C is a setpoint the card will find

Even at 100% fans, sustained single-card load reaches 79 °C at ~60 s and settles at
1215-1227 MHz. Turbo delays the throttle (30 s → 60 s) and raises the floor
(1113 → ~1220 MHz). It does not remove it. Ducting for passive cards is the
remaining ~8%; there is nothing further to win in software.

## Consequences for benchmarking

1. **Always start from a matched temperature.** An A/B where one side ran on a warm
   card is meaningless. This produced one completely wrong conclusion here before
   it was caught.
2. **Sample clocks and throttle reasons during the run, not after.** An idle sample
   shows `Idle: Active` and tells you nothing. A short burst can sit at full boost
   and hide a throttle that a real workload would hit within a minute.
3. **`utilization.gpu` at 100% does not mean the card is working hard.** During
   decode it reads 100% while the memory controller is 18-31% busy.

Useful, no root required:

    # throttle reasons and clocks, during the run
    nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,\
    clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown \
    --format=csv,noheader -i 0

    # chassis power on a Dell, without racadm/sudo
    grep -l '^power_meter$' /sys/class/hwmon/hwmon*/name | xargs dirname
    # then read power1_average (microwatts). The hwmon NUMBER IS NOT STABLE
    # across boots - look it up by name every time.
