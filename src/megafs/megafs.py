import os

import pandas as pd 
import numpy as np
import random


from mrmr import mrmr_classif

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

from xgboost.sklearn import XGBClassifier

from genetic_selection import GeneticSelectionCV


from typing import Dict, Optional
from sklearn.base import ClassifierMixin
from sklearn.base import is_classifier
from sklearn.metrics import get_scorer_names
from sklearn.metrics._scorer import _BaseScorer

def validate_dataset(dataset):
    """
    Validate the existence and structure of an Excel file with the dataset

    This function checks if the file exists, if it is readable by pandas, 
    if it contains the required 'class' column, and if all other columns 
    are numeric.

    Parameters
    ----------
    dataset : str
        The path or name of the Excel file to validate (e.g., 'data.xlsx').

    Returns
    -------
    pd.DataFrame 
        Returns the loaded DataFrame if all validations pass.
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist at the specified path.
    ValueError
        If the file is empty or cannot be parsed as an Excel file.
    KeyError
        If the required 'class' column is missing.
    TypeError
        If non-numeric data is found in feature columns.
    """
    print(f"File '{dataset}' Reading dataset....")
    # 1. Check if the file exists
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"The file '{dataset}' was not found.")
        
    try:
        # 2. Try to read the Excel file
        df = pd.read_excel(dataset, header=0)
    except Exception as e:
        raise ValueError(f"Failed to parse Excel file: {e}")
        
    if df.empty:
            raise ValueError("The DataFrame is empty. There is no data to process.")
            
        # 3. Check if 'class' column exists
    if 'class' not in df.columns:
        raise KeyError(f"Required column 'class' is missing in '{dataset}'.")

        # 4. Check if all columns are numeric
        # We use select_dtypes to find columns that are NOT numeric
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if len(non_numeric_cols) > 0:
        raise TypeError(f"Non-numeric columns found: {non_numeric_cols}")

    print(f"Success: Dataset '{dataset}' loaded and validated. \n")
    return df

def validate_classifier_dict(model_list):
    """
    Validates if the provided dictionary contains valid scikit-learn classifiers

    Parameters
    ----------
    classifiers: dict
        A dictionary mapping names (str) to potential classifier instances. 

    Returns:
        bool: True if all objects are valid classifiers.

    Raises:
        TypeError: If any value in the dictionary does not implement 
            the scikit-learn classifier interface.
    """
    if not isinstance(model_list, dict):
        raise TypeError(f"{model_list} is not a dictionary (name and model)")

    for model, model_inst  in model_list.items():
        if not is_classifier(model_inst):
           raise TypeError(f"Validation failed: '{model_inst}' is not a valid scikit-learn classifier. ")
    
    return True


def validate_sklearn_classifier_metric(metric):
    """
    Validates if the provided metric is a valid scikit-learn classification scorer.

    Parameters
    ----------
    metric : str or callable
        The name of the scorer (str) or a scikit-learn scorer object.

    Returns
    -------
    bool : True if the metric is valid.

    Raises
    ------
    ValueError : If the string metric is not in the list of valid scorers.
    TypeError : If the metric is neither a recognized string nor a valid scorer object.
    """
    # Get all valid scorer names (accuracy, f1, precision, etc.)
    valid_scorers = get_scorer_names()

    if isinstance(metric, str):
        if metric not in valid_scorers:
            raise ValueError(f"'{metric}' is not a valid scikit-learn metric. "
                             f"Choose from: {valid_scorers}")
        return True

    # Check if it's a scikit-learn scorer object (created via make_scorer)
    if isinstance(metric, _BaseScorer) or callable(metric):
        return True

    raise TypeError(f"Metric must be a string or a callable. Got {type(metric)} instead.")


#######################  MEGAFS #############################
def megafs(dataset, model_list = None, factor_maxFpI = 1, factor_nFiSE = 2, percent_feat_total = 1, 
          metric = 'roc_auc', n_splits = 5, n_runs = 3, output_file='Results_MEGAFS'):
    """
    Parameters
    ----------
    dataset: str
        The path or name of the Excel file (e.g., 'data.xlsx').
        Input dataset for classification. The structure must have individuals (samples) as rows 
          and genes (features) as columns. It must also include a specific column named 'class' containing the output labels
    model_list: dict
        A dictionary where keys are model names (str) and values are 
        scikit-learn estimator objects (e.g., {'RFC': RandomForestClassifier()}).
    factor_maxFpI: float
        Scaling factor used to multiply the F value 
            to to define the maxFpI parameter
        Defaults to 1.
    factor_nFiSE: float
        Scaling factor used to multiply the F value to define the nFiSE parameter. 
        Defaults to 2.
    percent_feat_total: float
        The percentage of total features to be processed. 
        Must be a value in the range (0, 1]. Defaults to 1.
    metric: str
        The name of the metric to be used for evaluating the models.
        Must be a valid scikit-learn classification metric (e.g., 'accuracy', 'f1', 'roc_auc').
        Defaults to 'roc_auc'.
    n_splits: int
        Number of fold into which the dataset is divided to cross-validation process. 
        Must be at least 2.
        Defaults to 5. 
    n_runs: int
        The number of complete, independent runs of the 
        non-deterministic algorithms. 
        Must be an integer between 1 and 10.
        Defaults to 3.
    output_file: str
        The name of output files.
        Defaults to 'Results_MEGAFS.txt' and 'Results_MEGAFS.xlsx'
        
        
    Output files
    -------
    output_file.txt: 
          The file contains the values of parameters used in the experimentation and for each model the following information:
            Model: name_model number of runs
            maxFpI: maximun value of features selected per individual
            n_features_selected: number of selected features
            roc_auc_Test: test result obtained
            roc_auc_Train: train result obtained
            Features_selected: list of selected featured
            Generation_Scores: the evolution of the fitness along the genetic algorithm training generation

    Result_MEGAFS_metrix.xlsx: This file contains the mean execution metric and the mean feature number 
                               for each model and run.

        
    """
    
    ############## Parameter Validation ##############
    
    print('Validating parameters ...')
    
    try:
        data = validate_dataset(dataset)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as e:
        print(f"ERROR: {e}")
        return
        
    
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    
    if model_list is None:
        # List used in the paper
        # model_list = {'AdaBC': AdaBoostClassifier(random_state=seed),
        #              'GBC': GradientBoostingClassifier(random_state=seed),
        #              'GNB': GaussianNB(),
        #              'GPC': GaussianProcessClassifier(random_state=seed),
        #              'KNNC': KNeighborsClassifier(),
        #              'LDA': LinearDiscriminantAnalysis(),
        #              'LR': LogisticRegression(random_state=seed),
        #              'MLPC': MLPClassifier(random_state=seed),
        #              'MNB': MultinomialNB(),
        #              'NuSVC': NuSVC(random_state=seed),
        #              'QDA': QuadraticDiscriminantAnalysis(),
        #              'RFC': RandomForestClassifier(random_state=seed),
        #              'SGDC': SGDClassifier(random_state=seed),
        #              'SVC': SVC(random_state=seed),
        #              'XGBC': XGBClassifier(random_state = seed)
        #              }
        

        model_list = {'GBC': GradientBoostingClassifier(random_state=seed),
                      'GNB': GaussianNB(),
                      'KNNC': KNeighborsClassifier(),
                      'MNB': MultinomialNB(),
                      'SGDC': SGDClassifier(random_state=seed),
                      'SVC': SVC(random_state=seed)
        }
    
    try:
        validate_classifier_dict(model_list)
    except (TypeError) as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")    
    else:
        print("Method list successfully validated. \n")
    
    
    if (percent_feat_total <= 0):
        print(f'percent_feat_total ({percent_feat_total}) must be greater than 0')
        print('percent_feat_total is set to 1 To process all features')
        percent_feat_total = 1
       
    
    if (n_splits < 2 or n_splits >=10):
        print(f'n_splits ({n_splits}) must be between 2 and 10')
        print('n_splits is set to 5 (default value)')
        n_splits = 5
        
    if (n_runs < 1 or n_runs > 10):
        print(f'n_runs ({n_runs}) must be between 1 and 10')
        print('n_runs is set to 35 (default value)')
        n_runs = 3
    
    ############## Split the dataset into input and output ##############
      
    X = data.drop(columns=['class'])
    y = data['class']
      
    ############## Dataset scaling ##############
    print('Dataset scaling')
    X_origen = X
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(X,columns=X_origen.columns)
    X_origen = X

    ############## mRMR selection ##############
           
    feat_total_selected = int(X.shape[1]*percent_feat_total)
    print(f'Selecting {feat_total_selected} features with mRMR')
    
    ordered_features = mrmr_classif(X = X, y = y, K = feat_total_selected)
    
    step = 1
    ini = 1
    r = range(ini,feat_total_selected+1, step)
    
    result_models_train = pd.DataFrame(index = r, columns= model_list.keys())
    result_models_test = pd.DataFrame(index = r, columns= model_list.keys())
    
    partitions = StratifiedKFold(n_splits = n_splits, shuffle =True, random_state=seed)
    
    
    for k in r:
        print (f'Training models with {k} features selected by mRMR')
                
        X = X_origen[ordered_features[0:k]]   
        
        for model, model_inst  in model_list.items():  
            print(f'  Training {model} model...')
            try:
                cv_results = cross_validate(model_inst, X, y, cv=partitions, return_estimator=True, 
                                        scoring=[metric],
                                        error_score='raise',
                                        return_train_score=True)
            except:
                print(f'ERROR when executing {model}')
                continue
           
            avg_accuracy_train = np.mean(np.abs(cv_results['train_'+metric]))
            avg_accuracy_test = np.mean(np.abs(cv_results['test_'+metric]))
                    
            result_models_train.at[k,model] = avg_accuracy_train
            result_models_test.at[k,model] = avg_accuracy_test
        
            
    result_models_test.loc['Max'] = result_models_test.max()
    result_models_test.loc['N_features'] = result_models_test.astype('float64').idxmax()+1         
    
    
    model_names = list(model_list.keys())   
    n_features_max = result_models_test.iloc[-1].values
        
    F_list = dict(zip(model_names, n_features_max))
    
        
    ############## Genetic Algorithm selection ##############
    print('Selecting features with genetic algorithm...')
    minFpI = 1
    
    f1 = output_file + ".txt"  
    with open(f1,"a") as f:                                   
        print(f'dataset = {dataset}; model_list = {model_list}; factor_maxFpI = {factor_maxFpI}; factor_nFiSE = {factor_nFiSE}; percent_feat_total = {percent_feat_total}; metric = {metric}; n_splits = {n_splits}; n_runs = {n_runs}', file = f)
        

    df_result = pd.DataFrame()  
    for model, model_inst  in model_list.items():   
        estimator = model_inst
        maxFpI =  F_list[model]*factor_maxFpI
        nFiSE = F_list[model] * factor_nFiSE
        
        if (nFiSE > feat_total_selected ):
            print(f'nFiSE({nFiSE}) must be less than or equal to the number of features processed with mRMR ({feat_total_selected})')
            print(f'nFiSE is set to {feat_total_selected}')
            nFiSE = feat_total_selected
        
        if (maxFpI > nFiSE):
            print(f'maxFpI({maxFpI}) must be less than or equal to nFiSE({nFiSE})')
            print(f'maxFpI is set to {nFiSE}')
            maxFpI = nFiSE
        
        print(f'FiSE = {nFiSE} and maxFpI = {maxFpI}')
        
        results_runs = []  
        for run in range(1, n_runs + 1): 
            print(f'Run {run} of {model}')
            X = X_origen[ordered_features[0:nFiSE]]   
            selector = GeneticSelectionCV(
                estimator,
                cv=partitions,
                verbose=1,
                scoring=metric,
                min_features=minFpI,
                max_features=maxFpI,
                n_population=300, 
                crossover_proba=0.5,
                mutation_proba=0.2,
                n_generations=100,
                crossover_independent_proba=0.5,
                mutation_independent_proba=0.05,
                tournament_size=3,
                n_gen_no_change=25,
                caching=True,
                #n_jobs=4
            )
            selector = selector.fit(X, y)
          
            X = X_origen[X.columns[selector.support_]]
            cv_results = cross_validate(model_inst, X, y, cv=partitions, return_estimator=True, 
                                        scoring=[metric],
                                        error_score='raise',
                                        return_train_score=True)
            avg_metric_train = np.mean(np.abs(cv_results['train_'+metric]))
            avg_metric_test = np.mean(np.abs(cv_results['test_'+metric]))
            
            with open(f1,"a") as f:                                   
                print(f'#### Model maxFpI n_features_selected {metric}_Test {metric}_Train Features_selected Generation_Scores ####', file = f)
                print(f'{model} run {run}', file = f)
                print(maxFpI, file = f)
                print(selector.n_features_, file = f)
                print(avg_metric_test, file = f)
                print(avg_metric_train, file = f)
                print(*X.columns, file = f)
                print(selector.generation_scores_, file = f)
            
        # end loop runs    
            results_runs.append(avg_metric_test)
            results_runs.append(selector.n_features_)
        
        df_result[model] = results_runs
    
    # end loop models
    row_names = []
    for run in range(1, n_runs + 1):
        row_names.append(f"{metric}_{run}")
        row_names.append(f"nFeatures_{run}")
                          
    avg_metric = df_result.iloc[::2].mean()
    avg_features = df_result.iloc[1::2].mean()
    df_result = df_result.rename(index=dict(enumerate(row_names)))
    df_result.loc['Mean_'+metric] = avg_metric
    df_result.loc['Mean_nFeatures'] = avg_features
    f2 = output_file+".xlsx";
    df_result.to_excel(f2)
    
    print(f'The results are shown in output files {output_file}.txt and {output_file}.xlsx')
