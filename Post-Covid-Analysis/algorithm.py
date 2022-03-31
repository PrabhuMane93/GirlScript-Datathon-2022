import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor_TeamDisrupt:
    
    # params_keys : ['learning_rate', 'n_estimators', 'max_depth', 'max_features']
    # params_keys : ['learning_rate', 'n_estimators', 'max_leaf_nodes', 'max_features', 'random_state', 'verbose']
    
    def __init__(self, params={}):
        # Hyperparameters of GradientBoost
        self.learning_rate = params.get('learning_rate', .1)
        self.n_estimators = params.get('n_estimators', 100)
        
        # Hyperparameters of Weak Regressor(DecisionTreeRegressor) of GradientBoost
        self.max_leaf_nodes = params.get('max_leaf_nodes', 8)
        self.max_features = params.get('max_features', None)
        
        # parameters for ...
        self.random_state = params.get('random_state', None)
        self.verbose = params.get('verbose', 0)
        
        self.fs = []    # stores all Decisiontree
    
    @staticmethod
    def del_loss(observed, predicted):
        return observed - predicted # return Residual
    
    @staticmethod
    def loss(observed, predicted):
        residual = GradientBoost.del_loss(observed, predicted)
        return np.dot(residual, residual.T)
        
    def fit(self, data, target):
        """
            data, target is instance of numpy.
            
            data.shape : ( no.entities, no.features )
            target.shape : ( no.entities, )
        """
        # STEP 1 : Init model with a constant value, F0(x) = argmin(r)(Σ(loss(y, r)))
        ## r is equal with average of all values of target
        r = np.average(target)
        f0 = self.F0(r)
        self.fs.append(f0)
        
        # STEP 2 : makes and fit Weak predictor (We use DecisionTree) for each step
        for i in range(1, self.n_estimators+1):
            if self.verbose == 1 and i%10 == 0: print("[NOTICE] FIT {0}-Weak Regressor".format(i))
            # (A) : Compute r_im
            # (B)
            residual = target - self.predict(data, n=i)
            cur_f = DecisionTreeRegressor(\
            max_leaf_nodes=self.max_leaf_nodes, max_features=self.max_features, random_state=self.random_state)
            cur_f.fit(data, residual)
            # (D) : Update predictor, Fm(x) = Fm-1(x) + vΣrjm I (x Rjm)
            self.fs.append(cur_f)
        if self.verbose == 1: print("[NOTICE] GRADIENT BOOST FIT FINISHED")
        
    def predict(self, data, n=None):
        if n is None : n = self.n_estimators
        ret = self.fs[0].predict(data)
        for i in range(1, len(self.fs)):
            ret += self.fs[i].predict(data) * self.learning_rate
        return ret
        
    class F0(DecisionTreeRegressor):
        def __init__(self, r):
            self.r = r
            pass
        
        def fit(self, data, target):
            pass
        
        def predict(self, data):
            return np.full(data.shape[0], self.r)

