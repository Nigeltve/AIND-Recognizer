import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM

from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """


    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        '''
        # note: this works but if you keep getting the error in the unittest
        #    'function' object has no attribute 'n_components'
        # probably something wrong with the unittest.
        
        
        # my implimentation 
        
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_bic = None

        for n_com in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm = GaussianHMM(n_components=n_com, covariance_type="diag",
                                  n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X,self.lengths)
                logL = hmm.score(self.X, self.lengths)

                p = n_com ** 2 + 2 * n_com * len(self.X[0]) - 1
                n = len(self.sequences)
                bic = -2 * logL + p * np.log(n)

                if best_bic is None:
                    best_bic = bic
                    best_model = hmm

                if bic < best_bic:
                    best_bic = bic
                    best_model = hmm
            except:
                pass



        return best_model
        '''

        #this is the code from the review.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score, best_n_components = None, None

        min_bic = float('inf')
        best_model = None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                N = sum(self.lengths)
                n_features = self.X.shape[1]
                p = n_components * (n_components - 1) + 2 * n_features * n_components
                # calculate BIC score
                bic = -2 * logL + p * logN
                if bic < min_bic:
                    min_bic = bic
                    best_model = model
            except Exception as e:
                continue
        return best_model if best_model else self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        M = len(self.words.keys())
        best_model = None
        best_dic = float('-inf')

        for n_component in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(num_states=n_component)
                logL = hmm_model.score(self.X, self.lengths)

            except:
                logL = float('-inf')

            log_sum = 0
            for each_word in self.hwords.keys():
                idx_word, word_lengths = self.hwords[each_word]

            try:
                log_sum += hmm_model.score(idx_word, word_lengths)

            except:
                log_sum += 0

            dic_score = logL - (1 / (M - 1)) * (log_sum - (0 if logL == float("-inf") else logL))

            if dic_score > best_dic:
                best_dic = dic_score
                best_model = hmm_model

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        max_score = None

        if len(self.lengths) == 1:

            for n_comp in range(self.min_n_components, self.max_n_components + 1):

                try:

                    Gaus_HMM = GaussianHMM(n_components=n_comp, covariance_type='diag', n_iter=4000,
                                           random_state=self.random_state, verbose=False)

                    model = Gaus_HMM.fit(self.sequences[0], self.lengths)
                    score = Gaus_HMM.score(self.X, self.lengths)

                    if max_score is None:
                        max_score = score
                        best_model = model

                    if score > max_score:
                        max_score = score
                        best_model = model
                except:
                    pass
            return best_model

        else:
            kf = KFold(n_splits=min(3, len(self.sequences)))

            for n_comp in range(self.min_n_components, self.max_n_components + 1):

                best_cv_model = None
                best_cv_score = None
                try:
                    for train_idx, test_idx in kf.split(self.sequences):

                        train_list, train_len = combine_sequences(train_idx, self.sequences)
                        test_list, test_len = combine_sequences(test_idx, self.sequences)

                        Gaus_HMM = GaussianHMM(n_components=n_comp, covariance_type='diag', n_iter=4000,
                                               random_state=self.random_state, verbose=False)
                        model = Gaus_HMM.fit(train_list, train_len)
                        score = Gaus_HMM.score(test_list, test_len)

                        if best_cv_score is None:
                            best_cv_model = model
                            best_cv_score = score

                        if score > best_cv_score:
                            best_cv_model = model
                            best_cv_score = score

                    if max_score is None:
                        max_score = best_cv_score
                        best_model = best_cv_model

                    if best_cv_score > max_score:
                        max_score = best_cv_score
                        best_model = best_cv_model

                except:
                    pass

                # print('Best CV Model: {}\n'
                #      'Best CV score: {}\n'.format(best_model, max_score))

            return best_model
