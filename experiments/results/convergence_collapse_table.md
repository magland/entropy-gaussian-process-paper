**lowpass 6 kHz / 30 kHz** ($\Delta = 0.25\sigma$, $\Delta/h_0 = 6.1$, $L = 33$, $n = 30000$):

| $N$ | estimate (bits/sample) | discarded replicates | lost-lock steps | mean ESS |
| --: | --: | --: | --: | --: |
| 300 | 7.54e+08 (collapsed) | 8/8 | 99.8% | 99% |
| 1000 | 7.49e+08 (collapsed) | 8/8 | 99.5% | 99% |
| 3000 | 5.76e+08 (collapsed) | 8/8 | 90.5% | 93% |
| 10000 | 2.1874 (provisional) | 5/8 | 36.3% | 31% |
| 30000 | 2.1817 (provisional) | 2/8 | 6.4% | 31% |
| 100000 | 2.1806 $\pm$ 0.0015 | 0/8 | 0.0% | 31% |

**Gaussian, $\tau=1.5$** ($\Delta = 0.5\sigma$, $\Delta/h_0 = 8.6$, $L = 13$, $n = 30000$):

| $N$ | estimate (bits/sample) | discarded replicates | lost-lock steps | mean ESS |
| --: | --: | --: | --: | --: |
| 300 | 2.73e+05 (collapsed) | 8/8 | 89.6% | 90% |
| 1000 | 1.6403 (provisional) | 7/8 | 54.5% | 42% |
| 3000 | 1.6380 $\pm$ 0.0016 | 0/8 | 0.0% | 42% |
| 10000 | 1.6360 $\pm$ 0.0018 | 0/8 | 0.0% | 42% |
| 30000 | 1.6353 $\pm$ 0.0018 | 0/8 | 0.0% | 42% |
