import matplotlib.pyplot as plt
import numpy as np
import bilby

NsplinesMax = 13
outdir = 'psd_only'
evidence = []
components = []
for j in range(3, NsplinesMax + 1):
    components.append(j)
    r = bilby.result.Result.from_hdf5(f"{outdir}/noise_splines{j}_result.h5")
    evidence.append(r.log_evidence)

evidence = np.array(evidence)
bayes_factors = evidence - evidence[0]
plt.plot(components, bayes_factors, '-x')
plt.ylabel("Bayes factor against 3-splines")
plt.xlabel("Number of components")
plt.tight_layout()
plt.savefig('bayes-factors')
