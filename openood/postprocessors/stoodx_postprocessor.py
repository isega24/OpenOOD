from typing import Any, Callable
from tqdm import tqdm
from openood.postprocessors import BasePostprocessor
import openood.utils.comm as comm
import pandas as pd
import os
import torch
from scipy import stats
import numpy as np
from crp.attribution import CondAttribution
from zennit.composites import Composite


class FeatureStractor(torch.nn.Module):
    '''
    Class for Feature Extraction
    
    A Feature Estractor object is used to extract features and their relevance from a specific layer of a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to extract features from.
    device : torch.device
        The device to use for the model.
    feature_name : str
        The name of layer representing the feature to extract.
    composite : zennit.composites.Composite, optional
        The composite to use for the attribution calculation. If not provided, the default composite will be used.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 feature_name:str,
                 atribut:bool,
                 composite:Composite = None,
                 relative=True
                 ):
        super(FeatureStractor, self).__init__()
        self.model = model.to(device)
        self.feature_name = feature_name
        self.atribution = CondAttribution(self.model,no_param_grad=True)
        self.composite = composite
        self.device = device
        self.atribut = atribut
        self.relative = relative
        

    '''
    Method to extract features and their relevance.
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to extract features from.
        
    Returns
    -------
    features : torch.Tensor
        The extracted features.
    relevance : torch.Tensor
        The relevance of the extracted features.
    '''
    def _atribution_calc(self,x:torch.Tensor):
        _x = x.to(self.device)
        _x.requires_grad = True
        x_class = self.model(_x)
        
        x_class = torch.argmax(x_class,dim=1).item()
        
        attr = self.atribution(
            _x,
            conditions=[{'y':[x_class]}],
            composite=self.composite,
            record_layer=[self.feature_name],
            init_rel=-1,
        )
        
        return attr

    '''
    Forward method for the model wraped
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to extract features from.
        
    Returns
    -------
    model_rediction : torch.Tensor
        The model prediction.
    '''
    def forward(self,x:torch.Tensor):
        return self.model(x)

    '''
    Call method for the model prediction
    '''
    def __call__(self,x:torch.Tensor):
        return self.forward(x)

    '''
    Method to extract the relative activations multiplied by the relevance  
    of the model for a specific input
    
    Parameters
    ----------
    x : torch.Tensor
        The input tensor to extract features from.
        
    Returns
    -------
    activations : torch.Tensor
        The relative activations of the model for the input.
    
    '''
    def features(self,x:torch.Tensor):
        if self.atribut:
            return self.atribute(x)
        else:
            return self.feature_activations(x)
    def feature_activations(self,x:torch.Tensor):
        attr = self._atribution_calc(x)
        importance_matrix = attr.activations[self.feature_name]
        if self.feature_name == "encoder" or self.feature_name == "features":
            # This is how ViT works for their feature space, I don't know why it is only used the first feature vector
            # of the final features. Reference in the forward function of the VisionTransformer class, line 12:
            # https://github.com/pytorch/vision/blob/ed55b0309fc3ed7d8abc4e4172b8a3c9852ef454/torchvision/models/vision_transformer.py#L301C20-L302C1
            importance_matrix = importance_matrix[:,0]
        if self.relative:
            importance_matrix = importance_matrix/torch.max(torch.abs(importance_matrix))
        return importance_matrix

    '''
    Method to extract the relative activations of the model for a specific input.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to extract features from.

        Returns
        -------
        activations : torch.Tensor
            The relative activations of the model for the input.
    '''
    def atribute(self,x:torch.Tensor):
        attr = self._atribution_calc(x)
        importance = attr.relevances[self.feature_name] #* attr.activations[self.feature_name]
        if self.feature_name == "encoder":
            # This is how ViT works for their feature space, I don't know why they only use the first feature vector
            # of the final features. Reference in the forward function of the VisionTransformer class, line 12:
            # https://github.com/pytorch/vision/blob/ed55b0309fc3ed7d8abc4e4172b8a3c9852ef454/torchvision/models/vision_transformer.py#L301C20-L302C1
            importance = importance[:,0]
        if self.relative:
            importance = importance/torch.max(torch.abs(importance))
        
        return importance


class STOODX:
    '''
    class for OOD Test detector. 

    Parameters
    ----------
    model : FeatureEstractor
        The model to test.
    distance :
        The distance function to use betwen the validation features and the test features. It must be a function that takes two torch.Tensors 
        and returns a torch.Tensor of shape (1,).
    ''' 
    def __init__(self, model:FeatureStractor,
                 distance:Callable = lambda x,y:torch.norm(x-y,dim=1),
                 k_neighbors:int = 50,
                 k_NNs:int = 50,
                 quantile:float = 0.99,
                 whole_test:bool = True,
                 ):
        self.model          = model
        self.distance       = distance
        self.k_neighbors    = k_neighbors
        self.k_NNs          = k_NNs
        self.quantile       = quantile
        self.whole_test     = whole_test
        
        self.feats          = None
        self.classes        = None
        self.feats_list = []
        self.classes_list = []

    def __call__(self,x:torch.Tensor)->torch.Tensor:
        '''
            Forward method for the model wraped
            
            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from.
                
            Returns
            -------
            model_prediction : torch.Tensor
                The model prediction.
        '''
        return self.forward(x)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        '''
            Forward method for the model wraped
            
            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from.
                
            Returns
            -------
            model_prediction : torch.Tensor
                The model prediction.
        '''
        return self.model(x)

    def addFeatures(self,x:torch.Tensor):
        '''
            Method to add features of x estracted from the model to features of the OODTest object. If the features are
            not initialized, the method will initialize them with the features of x. If not, it will add them.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from. Shape must be (B,...), where B is the batch size and 
                ... is the shape of the input accepted by the model.
            
            Returns
            -------
            None
        '''
        feats = self.model.features(x).squeeze()
        classes = self.forward(x).argmax(1).detach()
        self.feats_list.append(feats)
        self.classes_list.append(classes)

    def finalizeFeatures(self):
        '''
            Method to finalize the features and classes of the OODTest object by concatenating all the collected features and classes.

            Parameters
            ----------
            None
            
            Returns
            -------
            None
        '''
        if self.feats_list:
            self.feats = torch.cat(self.feats_list, dim=0)
            self.classes = torch.cat(self.classes_list, dim=0)
            self.feats_list = []
            self.classes_list = []

    def restartFeatures(self):
        '''
            Method to restart the features and classes of the OODTest object to None.

            Parameters
            ----------
            None
            
            Returns
            -------
            None
        '''
        self.feats = None
        self.classes = None
        self.feats_list = []
        self.classes_list = []

    def features(self,x:torch.Tensor)->torch.Tensor:
        '''
            Method to extract features of x from the model.

            Parameters
            ----------
            x : torch.Tensor
                The input tensor to extract features from. Shape must be (B,...), where B is the batch size and 
                ... is the shape of the input accepted by the model.
            
            Returns
            -------
            features : torch.Tensor
                The extracted features. Shape must be (B,F), where B is the batch size and F is the number of features.
        '''
        return self.model.features(x)

    def test(self,x:torch.Tensor,intraclass:bool=True)->pd.DataFrame:
        """
        Method to test if the input x is OOD.

        The algorithm will use the following steps:
        1. Calculate the class and features of the input x.
        2. Consider the subset of the validation features:
            1. If intraclass is True, consider the subset of the same class as x.
            2. Else, consider the whole validation set.
        3. If n_features given, process the features of the input x and the validation features subset:
            1. Obtain the features ordered in the validation features subset and choose the main n_features.
            The presence is calculated as the sum of features in the validation subset:
                $$
                presence(i) = \sum_{j=1}^{#{ValSet}} f(x_j)_i
                $$
            2. The less present features are changed to 0.
        4. Obtain the distances between the input x and the validation features subset with the distance function
            provided and choose the K nearest neighbors, $V = \{v_1,...,v_K\}$.
        5. For each of the K nearest neighbors, calculate the distance between this v_i and the rest of the validation
            features subset, $i \neq j$ and choose other its K nearest neighbors, $V_i = \{v_{i1},...,v_{iK}\}$.
        6. Calculate the following matrix:

            | d(v,v_1) | d(v,v_2) | ... | d(v,v_K) |
            | d(v_1,v^1_1) | d(v_1,v^1_2) | ... | d(v_1,v^1_K) |
            | ... | ... | ... | ... |
            | d(v_K,v^1_1) | d(v_K,v^1_2) | ... | d(v_K,v^1_K) |

            where v is the features of the input x, v_i is the validation features of the $i$-th nearest neighbor
            and v^i_j is the validation features of the $j$-th nearest neighbor of the $i$-th nearest neighbor of x.
        7. For each K-nearest neighbor, perform the Bayes test between the distance between x and the validation subset (first row
            of the matrix) and the distance between x_i and the rest of the validation subset (i-th row of the matrix).
            The Bayes test is performed with the implementation of the article :cite:`benavoli2017time`. The Bayes test needs a Region of Practical Equivalence (ROPE) and we set it to the 75% quantile of the differences between the distances of the validation subset.


        Parameters
        ----------
        x : torch.Tensor
            The input tensor to test. Shape must be (B,...), where B is the batch size and
            ... is the shape of the input accepted by the model.
        n_features : int, optional
            The number of features to use for the test. If not provided, all features will be used.
            The features used will be the ones with the most presence in the features set considered.
            Default is None.
        intraclass : bool, optional
            If True, the test will be performed on the subset of the validation features that belong to the same class
            as x is predicted to belong to by the model. If False, the test will be performed on the whole validation set.
            Default is True.

        Returns
        -------
        ood_scores : pd.DataFrame
            A DataFrame with the following columns:
            - "d(x,VAL)>d(x_i,VAL)": Bayes test probability that distance between x and the validation subset is greater
                than the distance between x_i and the rest of the validation subset.
            - "d(x,VAL)~d(x_i,VAL)": Bayes test probability that distance between x and the validation subset is equivalent
                to the distance between x_i and the rest of the validation subset.
            - "d(x,VAL)<d(x_i,VAL)": Bayes test probability that distance between x and the validation subset is less
                than the distance between x_i and the rest of the validation subset.
            The Datafrane will have as many rows as the number of K-nearest neighbors considered.

        """

        x_features  = self.features(x.unsqueeze(0)).squeeze(0).detach().flatten()
        x_class     = torch.argmax(self.model(x.unsqueeze(0)).squeeze(0),dim=0)

        if intraclass:
            feat_subset = self.feats[self.classes == x_class]
        else:
            feat_subset = self.feats

        sum_abs_features = torch.sum(torch.abs(feat_subset),dim=0)

        
        Quantil = torch.quantile(sum_abs_features,self.quantile).item()

        n_features = torch.sum(sum_abs_features >= Quantil).item()

        # Guardar n_features en un archivo de logs llamado n_features.log
        # with open(f"XAI_features/features_NN{self.k_neighbors}.log","a") as f:
        #    f.write(f"len()={len(sum_abs_features)} min()={torch.min(sum_abs_features).item()} max()={torch.max(sum_abs_features)} Selected={n_features}\n")
        least_present_idx = torch.argsort(sum_abs_features,descending=True)[n_features:]

        feat_subset[:,least_present_idx] = 0

        x_features[least_present_idx] = 0

        x_distances = self.distance(x_features,feat_subset)
        
        if self.k_neighbors == -1:
            sorted_x_distances_idx = torch.argsort(x_distances)
        else:
            sorted_x_distances_idx = torch.argsort(x_distances)[:self.k_neighbors]

        distances_knns = torch.zeros(
            (len(sorted_x_distances_idx), len(feat_subset)), device=feat_subset.device
        )
        for i in range(len(sorted_x_distances_idx)):
            distances_knns[i] = self.distance(
                feat_subset[sorted_x_distances_idx[i]], feat_subset
            )

        df_p_values = pd.DataFrame(columns=["p_value"],index= range(len(sorted_x_distances_idx)))
        distances_top_ks = torch.sort(distances_knns, dim=0).values[0 : len(sorted_x_distances_idx)]

        x_distances = x_distances[sorted_x_distances_idx][:len(sorted_x_distances_idx)].detach().cpu().numpy()

        for i in range(len(sorted_x_distances_idx[:self.k_NNs])):
            distances_i = distances_top_ks[:,i].detach().cpu().numpy()
            # rope = torch.quantile(distances_i, self.quantile).item() - torch.quantile(distances_i, 1-self.quantile).item()

            if self.whole_test:
                if np.sum(np.abs(x_distances - distances_i)) > 0:
                    bayes_test = stats.wilcoxon(
                        x_distances, 
                        distances_i,
                        alternative="greater",
                        nan_policy="omit",
                    )
                else:
                    # Devuelve un bayes test ficticio para que no de error
                    # con un p_valor de 1
                    
                    bayes_test = stats.wilcoxon(
                        x_distances, 
                        distances_i+1,
                        alternative="greater",
                        nan_policy="omit",
                    )
                    
            else:
                bayes_test = stats.wilcoxon(
                    x_distances-distances_i,
                    alternative="greater",
                    nan_policy="omit",
                )

            df_p_values.iloc[i] = bayes_test.pvalue

        return df_p_values




dists = {"cosine":lambda x,y:1 - torch.cosine_similarity(x, y,dim=1)}
class STOODXPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(STOODXPostprocessor, self).__init__(config)
        self.K = self.config.get('K',500)
        self.NNK = self.config.get('NN_K',500)
        self.distance = dists[self.config.get('distance',"cosine")]
        self.feature_name = self.config.get('feature_name',"encoder")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intraclass = self.config.get('intraclass',True)
        self.quantil = self.config.get('quantil',0.75)
        self.atribut = self.config.get('atribut',False)
        self.partition = self.config.get('partition','train')
        self.whole_test = self.config.get('whole_test',False)
        self.id_name = self.config.get("id_name")
        self.model_name = self.config.get("model_name")
        self.APS_mode = False
        self.oodTest = None
        def p_value_statistic(df:pd.DataFrame)->float:
            return df["p_value"].mean()
        self.p_value_statistic = config.get("p_value_statistic",p_value_statistic)

    def deleteIrrelevantFeatures(self,q:int=-1):
        if q != -1:
            new_feats = []
            classes_list   = []

            for classes in torch.unique(self.oodTest.classes):
                feats = self.oodTest.feats[self.oodTest.classes == classes]
                # shufle feats
                feats = feats[torch.randperm(len(feats))][:q]
                new_feats.append(feats)
                classes_list.append(torch.tensor([classes for i in range(len(feats))]))

            self.oodTest.feats      = torch.cat(new_feats).to(self.device)
            self.oodTest.classes    = torch.cat(classes_list).to(self.device)

    def setup(self, net: torch.nn.Module, id_loader_dict, ood_loader_dict):
        # Create the feature extractor
        net = net.to(self.device)
        feature_estractor = FeatureStractor(model = net, device=self.device, feature_name=self.feature_name,atribut=self.atribut).to(self.device)
        self.oodTest = STOODX(
            model=feature_estractor, distance=self.distance, quantile=self.quantil,
            whole_test=self.whole_test,k_neighbors=self.K,k_NNs=self.NNK
        )

        if os.path.exists(
            f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}.pth"
        ) and os.path.exists(
            f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}_classes.pth"
        ):

            self.oodTest.feats = torch.load(
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}.pth",
                weights_only=True,
                map_location=self.device,
            )
            self.oodTest.classes = torch.load(
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}_classes.pth",
                weights_only=True,
                map_location=self.device,
            )
            print("Features added before")
        elif self.oodTest.feats == None:
            if len(self.partition.split("_")) == 1:
                loader_dict = id_loader_dict[self.partition]
            else:
                # Une los loaders
                combined_dataset = [id_loader_dict[self.partition.split("_")[0]].dataset,
                                    id_loader_dict[self.partition.split("_")[1]].dataset]
                combined_dataset = torch.utils.data.ConcatDataset(combined_dataset)
                loader_dict = torch.utils.data.DataLoader(
                    combined_dataset,batch_size=32,shuffle=False
                    )
            for batch in tqdm(
                loader_dict, desc="Adding features..."
            ):
                data = batch['data'].to(self.device)

                self.oodTest.addFeatures(data)
                # liberar memoria de la gpu

            self.oodTest.finalizeFeatures()

            os.makedirs(f"./utils/features/",exist_ok=True)
            torch.save(
                self.oodTest.feats,
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}.pth",
            )
            torch.save(
                self.oodTest.classes,
                f"./utils/features/{self.id_name}_{self.model_name}_{self.feature_name}_{self.partition}_classes.pth",
            )

    def postprocess(self, net: torch.nn.Module, data:Any):

        pred_list = []
        confs = []
        for batch in tqdm(data,desc="Calculating OOD conf..."):
            data = batch['data'].to(self.device)
            pred = self.oodTest(data)
            pred_list.append(pred)

            for element in data:
                df_bayesianTest = self.oodTest.test(element,self.intraclass)
                bayesianTest = self.p_value_statistic(df_bayesianTest)

                confs.append(bayesianTest)

        preds = torch.cat(pred_list).argmax(dim=1).cpu().numpy().astype(int)
        confs = torch.tensor(confs).cpu().numpy()
        return preds, confs

    def inference(self, net, data_loader, progress = True):
        label_list = []
        preds, confs = self.postprocess(net, data_loader)

        for batch in tqdm(data_loader,desc="Calculating inference...",
                        disable=not progress or not comm.is_main_process()):
            label = batch["label"].to(self.device)
            label_list.append(label)

        labels = torch.cat(label_list).cpu().numpy().astype(int)

        return preds, confs, labels
