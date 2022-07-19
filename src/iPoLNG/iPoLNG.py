import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import numpy as np
from pyro.infer import SVI, TraceMeanField_ELBO
import time
from tqdm import trange
from statsmodels.regression.quantile_regression import QuantReg
from importlib_resources import files
import os

def get_params_iPoLNG(keys):
    Thetas_est = {k: torch.nn.functional.softmax(pyro.param("logTheta%s_loc" % str(i+1)).detach().cpu(),-1).numpy() for i,k in enumerate(keys)}
    alpha_ik = pyro.param("alpha0_ik").detach().cpu().numpy() 
    beta_ik = pyro.param("beta0_ik").detach().cpu().numpy()
    L_est = beta_ik / (alpha_ik+1)
    Ls_est_mode = {k: np.maximum((pyro.param("alpha%s_ik" % str(i+1)).detach().cpu().numpy() - 1)/pyro.param("beta%s_ik" % str(i+1)).detach().cpu().numpy(),0) for i,k in enumerate(keys)}
    return dict({"L_est":L_est, "Ls_est":Ls_est_mode, "Thetas_est":Thetas_est})

def clip_params_PoLNG():
    for param in ["alpha_ik","beta_ik"]:
        pyro.param(param).data.clamp_(min=1e-3)
        
def clip_params_iPoLNG(M):
    pyro.param("alpha0_ik").data.clamp_(min=1e-3)
    pyro.param("beta0_ik").data.clamp_(min=1e-3)
    for i in range(M):
        pyro.param("alpha%s_ik" % str(i+1)).data.clamp_(min=1e-3)
        pyro.param("beta%s_ik" % str(i+1)).data.clamp_(min=1e-3)
         
                
class PoLNG:
    def __init__(self,num_topics) -> None:
        self.num_topics = num_topics
        # define hyperparameters that control the prior
        self.alpha_k = torch.tensor(0.1)
        self.beta_k = self.alpha_k * self.num_topics
        # define parameters used to initialize variational parameters
        self.alpha_init = torch.tensor(0.5)
        self.beta_init = torch.tensor(0.5)
        self.mean_init = torch.tensor(0.0)
        self.sigma_init = torch.tensor(0.1)

    def model(self, W1):
        I1,J1 = W1.shape
        I = I1
        s1 = W1.sum(-1,keepdim=True)
            
        with pyro.plate("Theta1_plate", self.num_topics):
            # logTheta follows a logistic-normal distribution
            logTheta1_loc = torch.zeros(J1)
            logTheta1_scale = torch.ones(J1)
            logTheta1 = pyro.sample(
                "logTheta1", dist.Normal(logTheta1_loc, logTheta1_scale).to_event(1))
            theta1 = F.softmax(logTheta1, -1)

        with pyro.plate("data", I):
            # L follows a Gamma distribution
            L = pyro.sample("L", dist.Gamma(self.alpha_k, self.beta_k).expand([self.num_topics]).to_event(1))
            mean_obs1 = torch.matmul(s1*L, theta1)
            # the observed data follows a Poisson distribution
            pyro.sample("obs1", dist.Poisson(mean_obs1).to_event(1), obs=W1)

    def guide(self,W1):
        I1,J1 = W1.shape
        I = I1
        with pyro.plate("Theta1_plate", self.num_topics):
            logTheta1_loc_q = pyro.param("logTheta1_loc", lambda: self.mean_init*torch.zeros(self.num_topics,J1))
            logTheta1_scale_q = pyro.param("logTheta1_scale", lambda: self.sigma_init*torch.ones(self.num_topics,J1),constraint=dist.constraints.positive)
            pyro.sample("logTheta1", dist.Normal(logTheta1_loc_q, logTheta1_scale_q).to_event(1))
        with pyro.plate("data", I):
            alpha_k_q = pyro.param("alpha_ik", lambda: self.alpha_init*torch.ones(I,self.num_topics),constraint=dist.constraints.positive)
            beta_k_q = pyro.param("beta_ik", lambda: self.beta_init*torch.ones(I,self.num_topics),constraint=dist.constraints.positive)
            pyro.sample("L", dist.Gamma(alpha_k_q, beta_k_q).to_event(1))
                

class iPoLNG_helper:
    def __init__(self,num_topics,alpha0s,alpha_k=1,init=None) -> None:
        self.num_topics = num_topics
        # define hyperparameters that control the prior
        self.alpha_k = alpha_k
        self.beta_k =  (alpha_k+1)/self.num_topics
        self.alpha0s = torch.tensor(alpha0s)
        # define parameters used to initialize variational parameters
        self.alpha_init = torch.tensor(0.5)
        self.beta_init = torch.tensor(0.5)
        self.mean_init = torch.tensor(0.0)
        self.sigma_init = torch.tensor(0.1)
        self.init = init

    def model(self, W):
        Is = [Wi.shape[0] for Wi in W.values()]
        Js = [Wi.shape[1] for Wi in W.values()]
        # make sure the number of samples from different data modalities are the same 
        assert len(np.unique(Is))==1
        I = Is[0]
        s = [Wi.sum(-1,keepdim=True) for Wi in W.values()]
        logTheta, Theta = [], []    

        for i in range(len(W)):
            with pyro.plate("Theta%s_plate" % str(i+1), self.num_topics):
                # logTheta follows a logistic-normal distribution
                logTheta.append(
                    pyro.sample("logTheta%s" % str(i+1), dist.Normal(torch.zeros(Js[i]), torch.ones(Js[i])).to_event(1))
                )
                Theta.append(
                    F.softmax(logTheta[i], -1)
                )

        Ls = []    
        with pyro.plate("data", I):
            # L follows an inverse Gamma distribution
            L = pyro.sample("L", dist.InverseGamma(self.alpha_k, self.beta_k).expand([self.num_topics]).to_event(1))
            for i, Wi in enumerate(W.values()):
                # Li follows a Gamma distribution
                Ls.append(
                    pyro.sample("L%s" % str(i+1), dist.Gamma(self.alpha0s[i], self.alpha0s[i]/L).to_event(1))
                )
                # the observed data follows a Poisson distribution
                pyro.sample("obs%s" % str(i+1), dist.Poisson(torch.matmul(s[i]*Ls[i], Theta[i])).to_event(1), obs=Wi)
            

    def guide(self,W):
        Is = [Wi.shape[0] for Wi in W.values()]
        Js = [Wi.shape[1] for Wi in W.values()]
        # make sure the number of samples from different data modalities are the same 
        assert len(np.unique(Is))==1
        I = Is[0]
        logTheta_loc_q, logTheta_scale_q = [], []
        alpha0_max_pos = torch.argmax(self.alpha0s)
        for i in range(len(W)):
            with pyro.plate("Theta%s_plate" % str(i+1), self.num_topics):
                if (self.init is not None) and (i==alpha0_max_pos):
                    logTheta_loc_q.append(
                        pyro.param("logTheta%s_loc" % str(i+1), lambda: torch.tensor(self.init["logTheta_loc"]))
                    )
                    logTheta_scale_q.append(
                        pyro.param("logTheta%s_scale" % str(i+1), lambda: torch.tensor(self.init['logTheta_scale']),constraint=dist.constraints.positive)
                    )
                else:
                    logTheta_loc_q.append(
                        pyro.param("logTheta%s_loc" % str(i+1), lambda: self.mean_init*torch.zeros(self.num_topics,Js[i]))
                    )
                    logTheta_scale_q.append(
                        pyro.param("logTheta%s_scale" % str(i+1), lambda: self.sigma_init*torch.ones(self.num_topics,Js[i]),constraint=dist.constraints.positive)
                    )
                pyro.sample("logTheta%s" % str(i+1), dist.Normal(logTheta_loc_q[i], logTheta_scale_q[i]).to_event(1))

        with pyro.plate("data", I):
            alpha0_k_q = pyro.param("alpha0_ik", lambda: self.alpha_init*torch.ones(I,self.num_topics),constraint=dist.constraints.positive)
            beta0_k_q = pyro.param("beta0_ik", lambda: self.beta_init*torch.ones(I,self.num_topics),constraint=dist.constraints.positive)
            pyro.sample("L", dist.InverseGamma(alpha0_k_q, beta0_k_q).to_event(1))
            alphas_k_q, betas_k_q = [], []
            for i in range(len(W)):
                if self.init is not None:
                    alphas_k_q.append(
                        pyro.param("alpha%s_ik" % str(i+1), lambda: torch.tensor(self.init['alpha_ik']),constraint=dist.constraints.positive)
                    )
                    betas_k_q.append(
                        pyro.param("beta%s_ik" % str(i+1), lambda: torch.tensor(self.init['beta_ik']),constraint=dist.constraints.positive)
                    )
                else:
                    alphas_k_q.append(
                        pyro.param("alpha%s_ik" % str(i+1), lambda: self.alpha_init*torch.ones(I,self.num_topics),constraint=dist.constraints.positive)
                    )
                    betas_k_q.append(
                        pyro.param("beta%s_ik" % str(i+1), self.beta_init*torch.ones(I,self.num_topics),constraint=dist.constraints.positive)
                    )
                pyro.sample("L%s" % str(i+1), dist.Gamma(alphas_k_q[i], betas_k_q[i]).to_event(1))            
            
class iPoLNG:
    def __init__(self,W,num_topics,alpha_k=1,integrated_epochs=3000,warmup_epochs=3000,lr=0.1,seed=42,verbose=True) -> None:
        self.W = W
        self.num_topics = num_topics
        self.integrated_epochs = integrated_epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        # define hyperparameters that control the prior
        self.alpha_k = alpha_k
        self.beta_k =  (alpha_k+1)/self.num_topics
        # define parameters used to initialize variational parameters
        self.alpha_init = torch.tensor(0.5)
        self.beta_init = torch.tensor(0.5)
        self.mean_init = torch.tensor(0.0)
        self.sigma_init = torch.tensor(0.1)

    def Run(self):
        torch.manual_seed(self.seed)
        Is = [Wi.shape[0] for Wi in self.W.values()]
        # make sure the number of samples from different data modalities are the same 
        assert len(np.unique(Is))==1
        I = Is[0]
        opt = pyro.optim.Adam({"lr": self.lr, "betas": (0.90, 0.999)})
        starttime = time.time()
        # run PoLNG on single data modality
        PoLNGmodel = PoLNG(self.num_topics) 
        svi = SVI(PoLNGmodel.model, PoLNGmodel.guide, opt, loss=TraceMeanField_ELBO())        
        logTheta_loc, logTheta_scale, alphas_ik, betas_ik, losseses = [],[],[],[],[]
        for key, Wi in self.W.items():
            print('Run PoLNG for data modality {} ...'.format(key))
            pyro.clear_param_store()
            losses = []
            bar = range(self.warmup_epochs)
            if self.verbose:
                bar = trange(self.warmup_epochs)
            for _ in bar:
                running_loss = 0.0
                loss = svi.step(Wi)
                losses.append(loss)
                running_loss += loss / Wi.size(0)
                if self.verbose:
                    bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
                # clip the parameters to avoid frequent numerical errors
                clip_params_PoLNG()   
            logTheta_loc.append(
                pyro.param("logTheta1_loc").detach().cpu().numpy() 
            ) 
            logTheta_scale.append(
                pyro.param("logTheta1_scale").detach().cpu().numpy() 
            ) 
            alphas_ik.append(
                pyro.param("alpha_ik").detach().cpu().numpy() 
            ) 
            betas_ik.append(
                pyro.param("beta_ik").detach().cpu().numpy()
            )
            losseses.append(losses)
        # Ls_est_mode = [np.maximum((alphas_ik[i]-1)/betas_ik[i],0) for i in range(len(alphas_ik))]
        Ls_est_var = [alphas_ik[i]/betas_ik[i]**2 for i in range(len(alphas_ik))]
        Ls_est_mean = [alphas_ik[i]/betas_ik[i] for i in range(len(alphas_ik))]
        
        # perform quantile regression to get hyperparameters "alpha0s" that control the noise levels of data modalities
        self.alpha0s = [1 / QuantReg(Ls_est_var[i].reshape(-1),Ls_est_mean[i].reshape(-1,1)**2).fit(q=0.9).params[0] for i in range(len(alphas_ik))]
        for key, alpha0 in zip(self.W.keys(),self.alpha0s):
            print('alpha0 for data modality {}: {}'.format(key, alpha0))
        # run iPoLNG using the estimated alpha0s and initial values
        pyro.clear_param_store()
        print("Run iPoLNG...")

        alpha0_max_pos = np.argmax(self.alpha0s)
        init = dict({
            "logTheta_loc": logTheta_loc[alpha0_max_pos], 
            "logTheta_scale": logTheta_scale[alpha0_max_pos], 
            "alpha_ik": alphas_ik[alpha0_max_pos], 
            "beta_ik": betas_ik[alpha0_max_pos]
        })
        iPoLNGmodel = iPoLNG_helper(self.num_topics,alpha0s=self.alpha0s,alpha_k=self.alpha_k,init=init)
        svi = SVI(iPoLNGmodel.model, iPoLNGmodel.guide, opt, loss=TraceMeanField_ELBO())
        bar = range(self.integrated_epochs)
        if self.verbose:
            bar = trange(self.integrated_epochs)
        losses = []
        for _ in bar:
            running_loss = 0.0
            loss = svi.step(self.W)
            losses.append(loss)
            running_loss += loss / I
            if self.verbose:
                bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))
            # clip the parameters to avoid frequent numerical errors
            clip_params_iPoLNG(len(self.W))
        endtime = time.time()
        res = get_params_iPoLNG(self.W.keys())
        res['time'] = endtime - starttime 
        losseses = {key: loss for key,loss in zip(self.W.keys(),losseses)}
        losseses['iPoLNG'] = losses
        res['loss'] = losseses
        res['alpha0s'] = {key: alpha0 for key, alpha0 in zip(self.W.keys(),self.alpha0s)}
        pyro.clear_param_store()
        return(res)
        
def load_example_data():
    DATA_PATH = files('iPoLNG')
    W1=torch.tensor(np.load(os.path.join(DATA_PATH,"W1.npy")))
    W2=torch.tensor(np.load(os.path.join(DATA_PATH,"W2.npy")))
    return dict(W1=W1,W2=W2)