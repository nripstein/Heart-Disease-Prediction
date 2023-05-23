# This file is used to find the posterior pdf for sensitivity and specificity
# The code is more flexible than strictly needed for that task, because I reused this some from a more involved bayesian statistics project
# it would be faster if I redesigned it for this task (as I could easily avoid classes), but it's currently very fast

# imports
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from fractions import Fraction
import warnings


# custom errors
class PriorLikelihoodNumber(Exception):  # raised when there's a mismatch between number of priors and likelihoods
    pass


class HypothesisError(Exception):  # raised when there's an error with respect to hypotheses
    pass


class PriorError(Exception):  # raised when there's an error with respect to priors
    pass


class LikelihoodError(Exception):  # raised when there's an error with respect to likelihoods
    pass


class GeneralBayes:
    def __init__(self, likelihoods: Union[list, np.ndarray], priors: Union[list, np.ndarray] = None,
                 to_frac: bool = False, init_print: bool = True, latex_print: bool = False,
                 hypotheses: list = None, special_type: str = None) -> None:
        """
        a fairly flexible object which takes likelihoods and finds posteriors. assumes uniform prior unless specified
        otherwise.  posterior_pdf() function is great for plotting when not many hypotheses bc it labels everything well
        Can accept data as fractions or floats.  If all fractions, will return exact answers
        :param likelihoods: likelihoods as list or array
        :param priors: priors (optional, assumes uniform if not specified)
        :param to_frac: converts everything to fractions (can be imprecise with floating point arithmetic)
        :param init_print: print all posteriors nicely in init
        :param latex_print: print all posteriors as LaTeX in init
        :param hypotheses: list of hyp labels
        :param special_type: # not implemented yet. deals w weird stuff like 2d gaussian
        """

        if likelihoods is None:
            raise LikelihoodError(f"\nNo likelihoods provided.")

        if priors is None:
            priors = [Fraction(f"1/{len(likelihoods)}") for _ in likelihoods]

        # error catching
        if len(priors) != len(likelihoods):
            raise PriorLikelihoodNumber(f"\nDetected {len(priors)} priors and "
                                        f"{len(likelihoods)} likelihoods in GeneralBayes")
        if round(sum(priors), 5) != 1:
            raise PriorError(f"\nPriors must sum to 1. sum({priors}) = {sum(priors)} in GeneralBayes")

        if hypotheses is not None:
            if len(hypotheses) != len(likelihoods):
                raise LikelihoodError(f"\nlabelled hypotheses must have as many elements as likelihoods\n"
                                      f"detected {len(hypotheses)} hypotheses and {len(likelihoods)} likelihoods")
            else:
                self.hypotheses = hypotheses

        if special_type is not None:
            if special_type != "2d-gaussian":
                raise TypeError(f"\nValid special type is '2d-gaussian'.\n{special_type} is an invalid special type")

        for likelihood in likelihoods:
            if likelihood > 1:
                raise LikelihoodError(f"Likelihoods must be > 1. {likelihood} > 1 in GeneralBayes"
                                      f"\nLikelihoods = {likelihoods}")

        if not to_frac:
            self.priors: list = priors
            self.likelihoods: list = likelihoods
            self.fractions: bool = self._fraction_check()
        else:
            self.priors: list = [Fraction(i) for i in priors]
            self.likelihoods: list = [Fraction(i) for i in likelihoods]
            self.fractions: bool = True

        if latex_print:
            self.print_latex()
        elif init_print:
            print(self)

        self.__iter_index: int = 0
        self.arr: np.ndarray = self.posterior_array()

    def _fraction_check(self) -> bool:
        """Checks if all data is in fraction"""
        self.fractions: bool = False
        for i in self.priors:
            if type(i) != Fraction:
                return False
        for i in self.likelihoods:
            if type(i) != Fraction:
                return False
        return True

    def calculation(self, index):
        """Does 1 posterior calculation.  If all fractions, returns Fraction, else returns float"""
        likelihood = self.likelihoods[index]
        prior = self.priors[index]
        numerator = (likelihood * prior)
        denominator = 0
        for i in range(len(self.priors)):
            denominator += self.priors[i]*self.likelihoods[i]
        return numerator / denominator

    def posterior_array(self) -> np.ndarray[float]:
        output = np.zeros(len(self.priors))
        for i in range(len(self.priors)):
            output[i] = self.calculation(i)
        return output

    def posterior_list(self) -> list[Fraction]:  # if input is floats or ints, returns list of floats
        output = []
        for i in range(len(self.priors)):
            output.append(self.calculation(i))
        return output

    def mode_index(self):
        return np.argmax(self.arr)

    def posterior_pdf(self, save: bool = False, directory: str = "/Users/NoahRipstein/PycharmProjects/4kk3/plots",
                      file_name: str = "barplot_latest.png", title: str = "Posterior PDF", x_ticks: list = None,
                      x_label: str = None, top_numbers: bool = True):
        """
        :param save: save to png?
        :param directory: directory to save to
        :param file_name: name of file
        :param title: title above graph
        :param x_ticks: names of x ticks (hypothesis names)
        :param x_label: label on x-axis
        :param top_numbers: numbers on top of bars indicating P(H_n)?
        :return:
        """
        if x_label is None:
            x_label = "Hypotheses"

        fig, ax = plt.subplots()
        bars = ax.bar([f"H{i + 1}" for i in range(self.posterior_array().shape[0])], self.posterior_array(), edgecolor="black", alpha=0.8)

        if x_ticks is None and self.hypotheses is not None:
            x_ticks = self.hypotheses
        if x_ticks is not None:
            if len(x_ticks) < 20:
                ax.set_xticks(range(len(x_ticks)))
                # ax.set_xticklabels(x_ticks)
                ax.set_xticklabels([int(x) if x.is_integer() else x for x in x_ticks]) # makes into ints if thats what they are
            else:
                print("this isn't implimented yet. probably can get rid of this if else")
                # from matplotlib.ticker import MaxNLocator
                # ax.set_xticklabels(range(0, int(x_ticks.max()), 40))
                # ax.set_xticklabels([int(x) if x.is_integer() else x for x in x_ticks])
                # ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

        ax.set_xlabel(x_label)
        ax.set_ylabel("Posterior Probability")
        ax.set_title(title)

        if top_numbers:
            for i, bar in enumerate(bars):  # add labels on graph
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, "%.2f" % height, ha="center", va="bottom")

        if save:
            os.chdir(directory)
            plt.savefig(file_name)
        plt.show()
        return fig, ax

    def __getitem__(self, index: int) -> float:
        return self.arr[index]

    def __len__(self) -> int:
        return len(self.arr)

    def __str__(self) -> str:
        """Returns all posteriors to copy and paste"""
        output = f"{'-'*20}GENERAL BAYES POSTERIOR{'-'*20}\n"
        for index in range(len(self.priors)):
            if not self.fractions:
                output += f"P(H{index + 1}|D) = {self.calculation(index)}\n"
            else:
                current = self.calculation(index)
                output += f"P(H{index + 1}|D) = {self.calculation(index)} \u2245 {round((current.numerator / current.denominator) * 100, 2)}%\n"
        return output.rstrip()

    def print_latex(self) -> None:
        """Prints all posteriors as LaTeX to copy and paste"""
        output = ""
        for i in range(len(self.priors)):
            if not self.fractions:
                output += f"$ P(H{i + 1}|D) = {self.calculation(i)} $\\\\\n"
            else:
                current = self.calculation(i)
                output += f"$ P(H{i + 1}|D) = \\frac{{{current.numerator}}}{{{current.denominator}}} \doteq {round((current.numerator / current.denominator) * 100, 2)}\% $\\\\\n"
        print(output.rstrip())

    def __next__(self) -> float:     # this is needed for iter fn. returns next item
        arr = self.posterior_array()
        if self.__iter_index < arr.size:
            to_return = arr[self.__iter_index]
            self.iter_index = self.__iter_index + 1
            return to_return
        else:
            raise StopIteration

    def __iter__(self):     # works with __next__ to allow for loops to work
        return self


def hypothesis_dataframe(likelihoods: Union[list, np.ndarray], hypothesis_labels: Union[list, np.ndarray, tuple] = None,
                         priors: Union[list, np.ndarray] = None, prior_frac: bool = False,
                         export_to_general_bayes: bool = True) -> tuple[pd.DataFrame, Union[GeneralBayes, None]]:
    """
    given likelihoods, finds posteriors.  Assumes uniform priors, unless specified otherwise.  Can label hypotheses
    hypothesis_labels is tuple => first element should be 2d array, second should be names of hypotheses
    :param likelihoods: list or array of likelihoods
    :param hypothesis_labels: only be tuple for gaussian. element 1 is 2d array, element 2 is labels
    :param priors: assumes uniform. can be list or array
    :param prior_frac: True => creates priors as fractions
    :param export_to_general_bayes: False => returns (df, None). True => returns (df, GeneralBayes object)
    :return: dataframe with posteriors and my custom GeneralBayes object
    """

    if priors is None:
        if prior_frac:
            priors = [Fraction(f"1/{len(likelihoods)}") for _ in likelihoods]
        else:
            priors = [1/len(likelihoods) for _ in likelihoods]

    label_hypotheses: bool = False  # only needed to deal with labelling hypotheses in new cols

    if type(hypothesis_labels) == tuple:
        if type(hypothesis_labels[0]) == np.ndarray and len(hypothesis_labels[0].shape) == 2:
            hypothesis_array = hypothesis_labels[0].copy()
            hypothesis_specifics = list(hypothesis_labels[1])
            hypothesis_labels = None
            label_hypotheses = True
        else:
            raise HypothesisError(f"\nWhen hypothesis_labels is a tuple, first element must be 2d array")

    if hypothesis_labels is None:
        hypothesis_labels = [f"H{i + 1}" for i in range(len(likelihoods))]

    df = pd.DataFrame({"hypothesis": hypothesis_labels, "likelihood": likelihoods, "prior": priors})

    # this if else adds posteriors
    if df["prior"].nunique() == 1:  # if all priors are the same, don't bother w extra computations in else
        denominator = df["likelihood"].prod()
        l = df["likelihood"].to_numpy()
        df["posterior"] = l / np.sum(l)
    else:
        l = df["likelihood"].to_numpy()
        p = df["prior"].to_numpy()
        lxp = l * p
        posterior = lxp / np.sum(lxp)
        df["posterior"] = posterior

    if label_hypotheses:
        if len(hypothesis_specifics) != hypothesis_array.shape[1]:
            raise HypothesisError(f"\n Hypothesis labels needs to be the same shape as the hypotheses")
        for i in range(len(hypothesis_specifics)):
            df[hypothesis_specifics[i]] = hypothesis_array[:, i]

    if export_to_general_bayes:
        l = df["likelihood"]
        p = df["prior"]
        if label_hypotheses:  # might need to improve this for future. this isn't a clean way to estimate 2 parameters
            to_export = GeneralBayes(likelihoods=l, priors=p, hypotheses=df["hypothesis"], init_print=False, special_type="2d-gaussian")
        else:
            to_export = GeneralBayes(likelihoods=l, priors=p, hypotheses=df["hypothesis"], init_print=False)
        return df, to_export
    else:
        return df, None


def binomial_df(num_hypotheses: int, n: int, k: int, minimum: float = 0, maximum: float = 100,
                priors: Union[list, np.ndarray] = None) -> tuple[pd.DataFrame, GeneralBayes]:
    """
    :param num_hypotheses: number of hypotheses we should split into. aka number of bins
    :param n: larger than k. total number of guesses
    :param k: number that landed heads
    :param minimum: min hypothesis
    :param maximum: max hypothesis
    :param priors: optional priors
    :return: dataframe and general bayes object with the data
    """
    step = (maximum - minimum) / num_hypotheses
    using_min = minimum + (step / 2)
    using_max = maximum + (step / 2)
    hypotheses = np.arange(using_min, using_max, step)

    likelihoods_with_coefficient = stats.binom.pmf(k, n, hypotheses / 100)

    return hypothesis_dataframe(likelihoods_with_coefficient, hypothesis_labels=hypotheses)


def confidence(df: pd.DataFrame, value_col: str = "posterior", label_col: str = "hypothesis") -> tuple[float, float]:
    """
    finds 95% confidence interval
    :param value_col: name of column w values
    :param label_col: name of column w label
    :param df: dataframe with hypotheses and posteriors
    :return: minimum in interval, maximum in interval
    """
    df2 = df.sort_values(by=value_col, ascending=False)  # sorts df from highest to lowest posterior
    # print(df2.head(10))
    total = 0
    i = 0
    while total <= 0.95:  # finds the i largest elements which add up to 95%
        total += df2.iloc[i][value_col]
        i += 1
    df3 = df2.head(i - 1)  # makes dataframe with only i largest elements
    # [TO FIX] SHOULD BE i-1 BECAUSE I GOES UP HIGHER THAN NEEDED
    minimum = df3[label_col].min()
    maximum = df3[label_col].max()
    return minimum, maximum


def pdf_generator(correctly_labeled: int, mislabeled: int, xlab: str, title: str = "Posterior PDF",
                  hypotheses: int = 100, save: bool = True) -> tuple[float, tuple[float, float]]:
    """
    Sample usage:
    # sens_mode, (sens_95hi, sens_95hi) = sensitivity_specificity.pdf_generator(spam_result[0], spam_result[1], xlab="Spam Classifier Specificity (%)", title="Sensitivity PDF", hypotheses=500)
    # spec_mode, (spec_95hi, spec_95hi) = sensitivity_specificity.pdf_generator(ham_result[1], ham_result[0], xlab="Spam Classifier Sensitivity (%)", title="Specificity PDF", hypotheses=500)
    :param correctly_labeled: number correctly labelled
    :param mislabeled: number mislabelled
    :param xlab: x-axis label
    :param title: title of graph
    :param hypotheses: number of hypotheses
    :param save: if it should save as png
    :return: mode, (minimum value in 95% confidence interval, maximum value in 95% confidence interval)
    """
    os.chdir("/Users/NoahRipstein/PycharmProjects/Bayes email 2/visualizations")
    sensitivity_df, obj = binomial_df(hypotheses, correctly_labeled + mislabeled, correctly_labeled)
    min_95, max_95 = confidence(sensitivity_df)

    mode = sensitivity_df.iloc[obj.mode_index()]["hypothesis"]
    print(f"{xlab} mode = {mode}")
    print(f"{xlab} 95% confidence interval: {min_95} - {max_95}")

    fig, ax = plt.subplots(1, 1)
    ax.bar(range(len(sensitivity_df["posterior"])), sensitivity_df["posterior"], edgecolor="black", alpha=0.8)
    ax.set_xlabel(xlab)
    ax.set_ylabel("Posterior Probability")
    ax.set_title(title)

    # make it go from 0 to 100 instead of to 500
    ax.set_xlim([0, 100])
    # set the new x-axis tick positions and labels for each subplot
    new_tick_positions = np.linspace(0, 500, 6)
    new_tick_labels = np.linspace(0, 100, 6, dtype=int)
    ax.set_xticks(new_tick_positions)
    ax.set_xticklabels(new_tick_labels)

    if save:
        plt.savefig(f"{title}.png", dpi=300)
    plt.show()
    return mode, (min_95, max_95)


def sensitivity_specificity_pdfs(confusion_matrix: np.ndarray[(2, 2), int], hypotheses: int = 100, save: bool = True, title: str = "Sensitivity and Specificity PDFs") -> tuple[tuple[float, tuple[float, float]], tuple[float, tuple[float, float]]]:
    """
    makes pdf for sensitivity and specificity next to each other
    :param confusion_matrix: confusion matrix containing chart with correct and incorrect classifications. needs to be 2x2 of ints
    :param hypotheses: number of hypothesis bins
    :param save: if ture, saves as png
    :return: specificity mode, (min value in specificity 95% confidence interval, max value in specificity 95% confidence interval), same for sensitivity
    """

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))
    new_tick_labels = np.linspace(0, 100, 6, dtype=int)

    correctly_labeled_spam, mislabeled_spam = confusion_matrix[0]
    mislabeled_ham, correctly_labeled_ham = confusion_matrix[1]

    # sensitivity
    sensitivity_df, sensitivity_obj = binomial_df(hypotheses, correctly_labeled_spam + mislabeled_spam, correctly_labeled_spam)

    ax[0].bar(range(len(sensitivity_df["posterior"])), sensitivity_df["posterior"], edgecolor="dodgerblue")

    import seaborn as sns
    # sns.kdeplot(x=range(len(sensitivity_df["posterior"])), y=sensitivity_df["posterior"], ax=ax[0], fill=True)

    ax[0].set_xlabel("Specificity (%)")
    ax[0].set_ylabel("Posterior Probability")

    #  make sensitivity x axis go from 0 to 100 instead of to 500
    ax[0].set_xlim([0, 100])
    new_tick_positions = np.linspace(0, 1, 6) * len(sensitivity_df["posterior"])
    ax[0].set_xticks(new_tick_positions)
    ax[0].set_xticklabels(new_tick_labels)

    # specificity
    specificity_df, specificity_obj = binomial_df(hypotheses, correctly_labeled_ham + mislabeled_ham, correctly_labeled_ham)
    ax[1].bar(range(len(specificity_df["posterior"])), specificity_df["posterior"], edgecolor="dodgerblue")
    # sns.kdeplot(data=specificity_df, x="hypothesis", y="posterior", ax=ax[1], fill=True, bw_adjust=0.5)

    # print(specificity_df[specificity_df["posterior"]==specificity_df["posterior"].min()])

    ax[1].set_xlabel("Sensitivity (%)")

    #  make sensitivity x-axis go from 0 to 100 instead of to 500
    ax[1].set_xlim([0, 100])
    ax[1].set_xticks(new_tick_positions)
    ax[1].set_xticklabels(new_tick_labels)

    plt.tight_layout()
    fig.suptitle(title)
    plt.subplots_adjust(top=0.9)  # adjust the bottom margin
    if save:
        plt.savefig(f"Spam Classifier Sensitivity and Specificity.png", dpi=300)
    plt.show()

    sensitivity_min_95, sensitivity_max_95 = confidence(sensitivity_df)
    sensitivity_mode: float = sensitivity_df.iloc[sensitivity_obj.mode_index()]["hypothesis"]

    specificity_min_95, specificity_max_95 = confidence(specificity_df)
    specificity_mode: float = specificity_df.iloc[specificity_obj.mode_index()]["hypothesis"]

    return (sensitivity_mode, (sensitivity_min_95, sensitivity_max_95)), (specificity_mode, (specificity_min_95, specificity_max_95))

