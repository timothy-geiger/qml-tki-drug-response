# Timothy Geiger, acse-tfg22

from typing import List, Optional, Union

import numpy as np

# plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# custom imports
from ..data_handling.wrappers import DatasetWrapper

'''

Different Classes to make the plotting of different
specific plots easier that are frequently used in
especially EDA.

'''


class CorrMatrixPlotter:
    """
    Initialize the CorrMatrixPlotter.

    Args:
        dataset (DatasetWrapper): The dataset wrapper object.
    """
    def __init__(self, dataset: DatasetWrapper):
        self._dataset = dataset

        colors1 = plt.cm.OrRd_r(np.linspace(0, 1, 128))
        colors2 = plt.cm.BuGn(np.linspace(0, 1, 128))

        colorsCombined = np.vstack((colors1, colors2))
        self._negCmap = mcolors.LinearSegmentedColormap.from_list(
            'colormap', colorsCombined)

    def show(self,
             cols: List[str],
             show_labels: bool = True,
             size: tuple = (6, 6),
             calc: bool = True) -> None:
        """
        Display the correlation matrix plot.

        Args:
            cols (list[str]): The columns to include in the correlation matrix.
            show_labels (bool, optional): Whether to show the labels or not.
                Defaults to True.
            size (tuple, optional): The size of the plot.
            calc (bool): If the correlation should be calculated.

        Returns:
            None
        """

        _, ax = plt.subplots(figsize=size)

        self.heatmap = sns.heatmap(
            ax=ax,
            data=(self._dataset[cols].corr() if calc else self._dataset[cols]),
            annot=show_labels,
            cmap=self._negCmap,
            vmin=-1,
            vmax=1,
            xticklabels=True,
            yticklabels=True)

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.title("Correlation")
        plt.show()


class DistrubutionPlotter:
    """
    Initialize the DistrubutionPlotter.

    Args:
        dataset (DatasetWrapper): The dataset wrapper object.
    """
    def __init__(self, dataset: DatasetWrapper):
        self._dataset = dataset

    def show(self,
             cols: Union[str, List[str]]) -> None:
        """
        Display the distribution plot.

        Args:
            cols (str or list[str]): The column(s) to include in the
                distribution plot.

        Returns:
            None
        """

        if isinstance(cols, str):
            cols = [cols]

        data = self._dataset[cols]

        _, ax = plt.subplots(figsize=(8, 1*len(cols)))
        filtered_data = data[~data.isnull().any(axis=1)]

        sns.violinplot(data=filtered_data,
                       ax=ax,
                       palette='turbo',
                       inner=None,
                       linewidth=0,
                       saturation=0.4,
                       orient='h')

        sns.boxplot(data=filtered_data,
                    ax=ax,
                    palette='turbo',
                    width=0.3,
                    boxprops={'zorder': 2},
                    orient='h',
                    showmeans=True,
                    meanprops={
                        'marker': 'o',
                        'markerfacecolor': 'white',
                        'markeredgecolor': 'black',
                        'markersize': '8'})

        plt.show()


class FacetGridPlotter:
    """
    Initialize the FacetGridPlotter.

    Args:
        dataset (DatasetWrapper): The dataset wrapper object.
    """
    def __init__(self, dataset: DatasetWrapper):
        self._dataset = dataset

    def _label(self, x, color, label, label_pos):
        ax = plt.gca()

        if label_pos == 'right':
            x = 0.8
            y = .35

        elif label_pos == 'left':
            x = 0.2
            y = .35

        elif label_pos == 'middle':
            x = 0.5
            y = .35

        else:
            raise ValueError(label_pos + ' is not a valid ' +
                             'option.')

        ax.text(x, y, label, fontweight="bold", color=color,
                ha="center", va="center", transform=ax.transAxes,
                fontsize='medium')

    def show(self,
             cat_col: str,
             num_col: str,
             label_pos: str = 'right',
             cat_order: Optional[List[str]] = None) -> None:
        """
        Display the FacetGrid plot.

        Args:
            cat_col (str): The categorical column for the rows and hue.
            num_col (str): The numerical column for the plot.
            label_pos (str, optional): The position of the labels. Defaults to
                'right'.
            cat_order (list[str], optional): The order of categories. Defaults
                to None.

        Returns:
            None
        """

        sns.set_theme(
            style="white",
            rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=2.8)

        g = sns.FacetGrid(
            self._dataset._df,
            row=cat_col,
            hue=cat_col,
            aspect=8,
            height=2.6,
            row_order=cat_order,
            hue_order=cat_order)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, num_col, bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5)

        g.map(sns.kdeplot, num_col, clip_on=False, color="w",
              lw=2, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=-0.0001, linewidth=2, linestyle="-",
                  color=None, clip_on=False)

        g.map(self._label, num_col, label_pos=label_pos)

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel='')
        g.set(xlabel=num_col)
        g.despine(bottom=True, left=True)

        plt.show()

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)


class StackedBarPlotter:
    """
    Initialize the StackedBarPlotter.

    Args:
        dataset (DatasetWrapper): The dataset wrapper object.
    """
    def __init__(self, dataset: DatasetWrapper):
        self._dataset = dataset

    def show(self,
             col_x: str,
             col_y: str) -> None:
        """
        Display the stacked bar chart.

        Args:
            col_x (str): The column for the x-axis.
            col_y (str): The column for the y-axis.

        Returns:
            None
        """

        # create df
        tmp_df = self._dataset.groupby([col_x, col_y])\
            .size().reset_index().pivot(columns=col_x, index=col_y, values=0)

        # iterate over rows
        for index_row, row in tmp_df.iterrows():
            sumAll = row.sum()

            # convert to percentage
            for index_col, value in row.items():
                tmp_df.loc[index_row, index_col] = (value / sumAll) * 100

        # Plot
        tmp_df.plot(kind='barh',
                    stacked=True,
                    figsize=(10, 1 * tmp_df.shape[0]))

        plt.title("Stacked bar chart")
        plt.ylabel(col_y)
        plt.xlabel(col_x + " Percentage (%)")

        plt.legend(
            title=col_x, fontsize='medium', fancybox=True,
            loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)

        plt.show()
