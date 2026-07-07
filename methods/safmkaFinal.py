import numpy as np
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from bciflow.modules.fs.mibif import MIBIF
import numpy as np
# ----------------------------
# Utilitários
# ----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def clip_kernel(W):
    return np.clip(W, -1.0, 1.0)
def class_scatter(X):
    """
    Calcula o scatter de uma classe:
    S_k = sum_i (x_i - mu)(x_i - mu)^T
    """
    mu = X.mean(axis=0)
    S = np.zeros((X.shape[1], X.shape[1]))

    for x in X:
        diff = (x - mu).reshape(-1, 1)
        S += diff @ diff.T

    return S, mu
def within_class_scatter(X1, X2):
    S1, mu1 = class_scatter(X1)
    S2, mu2 = class_scatter(X2)
    SW = S1 + S2
    return SW, mu1, mu2
def between_class_scatter(mu1, mu2):
    diff = (mu2 - mu1).reshape(-1, 1)
    SB = diff @ diff.T
    return SB
def fisher_criterion(theta, SW, SB):
    num = theta.T @ SB @ theta
    den = theta.T @ SW @ theta
    return num / den
def fisher_direction(X1, X2, reg=1e-6):
    """
    Implementa exatamente:
    theta* ∝ SW^{-1} (mu2 - mu1)
    """
    SW, mu1, mu2 = within_class_scatter(X1, X2)

    # Regularização numérica (necessária em EEG)
    SW += reg * np.eye(SW.shape[0])

    theta = np.linalg.solve(SW, mu2 - mu1)

    return theta
def project(X, theta):
    return X @ theta
# ----------------------------
# Proposta Gaussiana por coeficiente
# ----------------------------
def propose_W_gaussian(Ws, T, sigma0=0.05, eps=1e-3, random_state=None):
    """
    W: array of coefficients, shape arbitrary (e.g. (n_elec, k) or (k,))
    T: temperatura atual (float >= 0)
    sigma0: base scale (tune)
    eps: evita sigma=0 para coeficientes exatamente 0
    Proposta: W_new = W + N(0, sigma_i)
      sigma_i = sigma0 * (|W_i| ) * sqrt(T)
    sqrt(T) -> diminui amplitude com queda de T de forma contínua.
    """
    rng = np.random.default_rng(random_state)
    for i in range(Ws.shape[0]):
        scale = sigma0 * (np.abs(Ws[i]) + eps) * np.sqrt(max(T, 1e-12))
        noise = rng.normal(loc=0.0, scale=scale)
        Ws[i] += noise

    return Ws

def compute_features(X, y, W, eps=1e-10):
    """
    X : (n_samples, n_bands, n_elec, n_time)
    y : (n_samples,)
    W : (kernel_len,) or (n_kernels, kernel_len)

    Retorna:
        F : feature matrix (n_samples, n_features)
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    n_samples, n_bands, n_elec, _ = X.shape

    # Garante W 2D: (n_kernels, kernel_len)
    if W.ndim == 1:
        W = W[None, :]
    n_kernels, _ = W.shape

    # --------------------------------
    # 1. Extração de log-power features
    # --------------------------------
    features = []

    for i in range(n_samples):
        sample_feats = []

        for b in range(n_bands):
            for k in range(n_kernels):
                kernel = W[k]
                logpowers = np.zeros(n_elec)

                for e in range(n_elec):
                    filtered = np.convolve(
                        X[i, b, e, :],
                        kernel,
                        mode="valid"
                    )
                    logpowers[e] = np.log(
                        np.mean(filtered ** 2) + eps
                    )

                sample_feats.append(logpowers)

        # concatena (band × kernel × elec)
        features.append(np.array(sample_feats))

    F = np.array(features).reshape(n_samples,n_bands*n_kernels,n_elec)  # (n_samples, n_features)
    # --------------------------------
    # 2. Seleção de features (opcional)
    # --------------------------------
    if F.shape[1] > 1:
        eegdata_fs = {"X": F, "y": y}
        mibif = MIBIF(8, clf=lda())
        mibif.fit(eegdata_fs)
        eegdata_fs = mibif.transform(eegdata_fs)
        F = eegdata_fs["X"].reshape(n_samples, -1)  # garante 2D para sklearn
    else:
        F = F.reshape(n_samples, -1)  # garante 2D para sklearn

    return F


def compute_energy(X, y, W, kernel_axis=2, eps=1e-10):
    """
    X: array shape (n_samples, n_elec, n_time) OR (n_samples, n_time)
    y: binary labels {0,1} shape (n_samples,)
    W: kernel (filter)
    alpha: peso do termo de separabilidade
    Retorna: energia (mean logloss + separability term)
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n_samples,n_bands, n_elec, time = X.shape

    F = compute_features(X, y, W, eps=eps)
    # ----------------------------
    # 2. LDA do sklearn
    # ----------------------------
    #separate 80% train to fit LDA
    f_train = F[:int(0.8*n_samples)]
    y_train = y[:int(0.8*n_samples)]
    # remaining 20% for energy computation
    F_test = F[int(0.8*n_samples):]
    y_test = y[int(0.8*n_samples):]
    clf = lda()
    clf.fit(f_train, y_train)
    y_pred = clf.predict_proba(F)[:,1]

    # y_pred = np.array([0 if y_pred[i,0]>0.5 else 1 for i in range(len(y_pred))])

    #calculate the classes in y_pred if value >=0.5 class 1 else class 0
    
    loss = - (y * np.log(y_pred+eps) + (1 - y) * np.log(1 - y_pred+eps))
    energy = loss.mean()

    
    # ----------------------------
    # 3. Termo de separabilidade SW/SB
    # ----------------------------
    class0 = F[y == 0] 
    class1 = F[y == 1]

    theta_star = fisher_direction(class0, class1)

    # Avalia critério de Fisher
    SW, mu1, mu2 = within_class_scatter(class0, class1)
    SB = between_class_scatter(mu1, mu2)

    separability_term = fisher_criterion(theta_star, SW, SB)


    return energy, separability_term

















# ----------------------------
# Simulated annealing principal (sem reset)
# ----------------------------
def anneal_find_W(eegdata, W0s,
                  n_iters=2000,
                  T0=1.0,
                  alpha=0.995,
                  sigma0=0.05,
                  eps_sigma=1e-3,
                  clip_kernel_flag=True,
                  random_state=None,
                  return_trace=False,
                  a = 1,b=0):
    """
    X, y: data
    W0: initial kernel (ndarray)
    n_iters: number of iterations
    T0: initial temperature
    alpha: multiplicative cooling per iter (T = T * alpha)
    sigma0, eps_sigma: proposal scales
    clip_kernel_flag: clip proposals to [-1,1]
    return_trace: if True, returns trace of energies
    """
    y = eegdata['y'].copy()
    X = eegdata['X'].copy()


    rng = np.random.default_rng(random_state)
    W = W0s.copy()
    W = clip_kernel(W) if clip_kernel_flag else W

    T = T0

    loglossFirst,separationFirst = compute_energy(X, y, W)

    E_prop = a*loglossFirst/separationFirst + b*separationFirst/separationFirst 
    E = E_prop

    W_best = W.copy()
    E_best = E

    best_sep_value = separationFirst/separationFirst
    best_logloss_value = loglossFirst/separationFirst
                

    energies = [E_prop]
    best_energies = [E_best]
    best_logloss = [best_logloss_value]
    best_sep = [best_sep_value]


    Ws = [W_best]

    for it in range(n_iters):
        # propose new W via gaussian per-coef
        W_prop = propose_W_gaussian(W, T, sigma0=sigma0, eps=eps_sigma, random_state=rng)
        if clip_kernel_flag:
            W_prop = clip_kernel(W_prop)

        logloss, separation  = compute_energy(X, y, W_prop)
        logloss = logloss/loglossFirst
        separation = separation/separationFirst
        # print("Iteration:", it+1, "Temperature:", T, "Current logloss:", logloss, "Current separation:", separation)

        E_prop = a*logloss + b*separation

        dE = E_prop - E
        accept = False
        if dE <= 0:
            accept = True
        else:
            # Metropolis criterion
            p_accept = np.exp(-dE / max(T, 1e-12))
            if rng.random() < p_accept:
                accept = True

        if accept:
            W = W_prop
            E = E_prop 
            # update best
            if E < E_best:
                E_best = E
                # print(E_best)
                W_best = W.copy()
                best_logloss_value = logloss
                best_sep_value = separation
                # -------- Validação PASSIVA --------
    


        # cool
        T = T * alpha

        # store trace optionally
        if return_trace: #and (it % max(1, n_iters // 200) == 0):
            energies.append(E_prop)
            best_energies.append(E_best)
            Ws.append(W_best.copy())
            best_logloss.append(best_logloss_value)
            best_sep.append(best_sep_value)

          

    print("Final Energy:", E_best)
    if return_trace:
        return W_best,energies,best_energies,best_logloss,best_sep, Ws
    else:
        return W_best, E_best




class safka_filter:
    
    def __init__(self,subject=1,num_kernels = 1,itrs=500,a=1,b=0,alpha=0.995,sigma0=0.05,kernel_size=65,collect_traces=False,seed=42):
        self.itrs = itrs
        self.a = a
        self.b = b
        self.alpha = alpha
        self.sigma0 = sigma0
        self.kernel_size = kernel_size
        self.best_loss = 0
        self.collect_traces = collect_traces
        self.seed = seed
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        W0s = []
        for i in range(num_kernels):
            W0 = rng.normal(scale=0.99, size=(kernel_size))
            W0 = clip_kernel(W0)
            W0s.append(W0)
        self.ws = np.array(W0s)
        self.bestWs = self.ws.copy()
        self.explorationEnergy = {f"subject_{subject}": [],} 
        self.bestEnergy = {f"subject_{subject}": [],} 
        self.wTrace = {f"subject_{subject}": [],} 
        self.bestLogLoss = {f"subject_{subject}": [],} 
        self.bestSeparation = {f"subject_{subject}": [],}



    def fit(self, eegdata):
        """
        Fit the kernel using safka algorithm.
        """
        print(f"Running safka with {self.itrs} iterations...")
        if self.collect_traces:
            print("Collecting energy traces during training...")
            start_time = time.time()
            self.bestWs,energies,benergies,loglosses,separations,ws = anneal_find_W(eegdata, self.ws,
                                    n_iters=self.itrs,
                                    T0=1.0,
                                    alpha=self.alpha,
                                    sigma0=self.sigma0,
                                    eps_sigma=1e-3,
                                    clip_kernel_flag=True,
                                    random_state=self.seed,
                                    return_trace=self.collect_traces,a=self.a,b=self.b)
            end_time = time.time()
            print(f"safka training completed. Time taken: {(end_time-start_time):.2f} seconds")
            self.explorationEnergy[f"subject_{eegdata['subj']}"].append(energies)
            self.bestEnergy[f"subject_{eegdata['subj']}"].append(benergies)
            self.wTrace[f"subject_{eegdata['subj']}"].append(ws)
            self.bestLogLoss[f"subject_{eegdata['subj']}"].append(loglosses)
            self.bestSeparation[f"subject_{eegdata['subj']}"].append(separations)

        else:
            start_time = time.time()
            self.bestWs,energies = anneal_find_W(eegdata, self.ws,
                                    n_iters=self.itrs,
                                    T0=1.0,
                                    alpha=self.alpha,
                                    sigma0=self.sigma0,
                                    eps_sigma=1e-3,
                                    clip_kernel_flag=True,
                                    random_state=self.seed,
                                    return_trace=self.collect_traces,a=self.a,b=self.b)
            end_time = time.time()
            print(f"safka training completed. Time taken: {(end_time-start_time):.2f} seconds")
        return self

    def transform(self, eegdata):
        """
        Apply the learned temporal kernels to EEG data.

        Input:
            X: (n_trials, n_bands, n_channels, n_times)

        Output:
            Z: (n_trials, n_bands * n_kernels, n_channels, n_times)
        """
        X = eegdata['X'].copy()

        n_trials, n_bands, n_channels, n_times = X.shape

        # Garante forma 2D: (n_kernels, kernel_len)
        if self.bestWs.ndim == 1:
            bestWs = self.bestWs[np.newaxis, :]
        else:
            bestWs = self.bestWs

        n_kernels, _ = bestWs.shape

        # Saída: concatenação banda × kernel
        Z = np.zeros((n_trials, n_bands * n_kernels, n_channels, n_times))

        for t in range(n_trials):
            for b in range(n_bands):
                for k in range(n_kernels):
                    out_idx = b * n_kernels + k
                    for c in range(n_channels):
                        Z[t, out_idx, c, :] = np.convolve(
                            X[t, b, c, :],
                            bestWs[k],
                            mode="same"
                        )

        eegdata['X'] = Z
        return eegdata

    def fit_transform(self, eegdata):
        self.fit(eegdata)
        return self.transform(eegdata)


# ----------------------------
# Exemplo de uso (pseudo-dados)
# ----------------------------
if __name__ == "__main__":
    # cria dados sintéticos para testar
    rng = np.random.default_rng(42)
    n_samples = 200
    n_bands = 1
    n_elec = 4
    n_time = 200
    X = rng.normal(size=(n_samples,n_bands, n_elec, n_time))

    # labels: metade 0 metade 1
    y = np.zeros(n_samples, dtype=int)
    y[n_samples // 2 :] = 1

    # inicializa kernel por eletrodo (ex: k=25)
    k = n_time//4  
    W0s = []
    for i in range(1):
        W0 = rng.normal(scale=0.2, size=(k))
        W0 = clip_kernel(W0)
        W0s.append(W0)
    W0s = np.array(W0s)

    eegdata = {'X': X, 'y': y, 'subj': 1}  

    W_best, E_best = anneal_find_W(eegdata, W0s,
                                   n_iters=50,
                                   T0=1.0,
                                   alpha=0.995,
                                   sigma0=0.04,
                                   eps_sigma=1e-3,
                                   clip_kernel_flag=True,
                                   random_state=123,
                                   a=0.5,b=0.5)
    print("Energia melhor:", E_best)
    print("W_best shape:", W_best.shape)
