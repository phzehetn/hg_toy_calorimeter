import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit

from plotting.general_2d_plot import  General2dBinningPlot


class ResponsePlot(General2dBinningPlot):
    def __init__(self,**kwargs):
        super(ResponsePlot, self).__init__(**kwargs)

    def draw(self, name_tag_formatter=None, return_fig=False, ax=None, **kwargs):
        fig = super().draw(name_tag_formatter, return_fig=True, ax=ax, **kwargs)
        axes = fig.axes[0] if ax is None else ax
        axes.axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        # axes.axhline(0, 0, 1, ls='--', linewidth=0.5, color='gray')
        if return_fig:
            return fig

    def _compute(self, x_values, y_values, weights=None):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        error = []

        lows = []
        highs = []

        bug_in_error_comptuation = True

        if weights is None:
            bug_in_error_comptuation = False
            weights = np.ones_like(y_values)


        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]


            filter = np.argwhere(np.logical_and(x_values >= l, x_values < h))
            filtered_y_values = y_values[filter].astype(float)
            filtered_weights = weights[filter].astype(float)

            m = np.sum(filtered_y_values*filtered_weights)/np.sum(filtered_weights)
            mean.append(m)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(m / np.sqrt(float(len(filtered_y_values))))

        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        return processed_data

def gauss_fnc(x, mu, a, sigma):
    return a*np.exp(-0.5*((x-mu)**2)/(sigma**2))

class ResponseByFitPlot(ResponsePlot):
    def __init__(self, debug_path=None, **kwargs):
        super().__init__(**kwargs)
        self.debug_path=debug_path

    def _compute(self, x_values, y_values, weights=None):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        error = []

        lows = []
        highs = []

        bug_in_error_comptuation = True

        if weights is None:
            bug_in_error_comptuation = False
            weights = np.ones_like(y_values)


        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]


            filter = np.argwhere(np.logical_and(x_values >= l, x_values < h))
            filtered_y_values = y_values[filter].astype(float)
            filtered_weights = weights[filter].astype(float)

            m = np.sum(filtered_y_values*filtered_weights)/np.sum(filtered_weights)
            # mean.append(m)

            range_ = np.min(filtered_y_values), np.max(filtered_y_values)
            y, bins = np.histogram(filtered_y_values, range=range_,
                                   bins=int(np.sqrt(len(filtered_y_values))),
                                   weights=filtered_weights,
                                   density=False)
            bins_ = bins
            bins = (bins[1:] + bins[0:-1]) / 2

            fitted, cov = curve_fit(gauss_fnc, bins, y,
                                    p0=(np.mean(filtered_y_values), np.max(y), np.std(filtered_y_values)),
                                    sigma=np.sqrt(y),
                                    absolute_sigma=True
                                    )
            e = np.sqrt(np.diag(cov))
            e = e[0]

            mu, a, sigma = fitted
            print("XYZ 2", range_,l,h, mu, len(filtered_y_values), np.min(x_values[filter]), np.max(x_values[filter]))



            # print('x', m, mu)
            # m = mu
            mean.append(mu)

            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(e)

        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        return processed_data


class ResolutionByFitPlot(General2dBinningPlot):
    def __init__(self,debug_path=None,**kwargs):
        super(ResolutionByFitPlot, self).__init__(**kwargs)
        self.debug_path = debug_path

    def draw(self, name_tag_formatter=None, return_fig=False, ax=None, **kwargs):
        fig = super().draw(name_tag_formatter, return_fig=True, ax=ax, **kwargs)
        axes = fig.axes[0] if ax is None else ax
        # axes[0].axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        axes.axhline(0, 0, 1, ls='--', linewidth=0.5, color='gray')
        if return_fig:
            return fig

    def add_raw_values(self, x_values, y_values, tags={}, weights=None):
        if type(x_values) is not np.ndarray:
            raise ValueError("x values has to be numpy array")
        if type(y_values) is not np.ndarray:
            raise ValueError("y values has to be numpy array")
        if weights is not None:
            if type(weights) is not np.ndarray:
                raise ValueError("weights has to be numpy array")

        data = self._compute(x_values, y_values, weights=weights, tags=tags)
        data['tags'] = tags
        self.models_data.append(data)

    def _compute(self, x_values, y_values, weights=None, tags={}):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []

        lows = []
        highs = []
        error = []

        bug_in_error_comptuation=True

        if weights is None:
            bug_in_error_comptuation=False
            weights = np.ones_like(y_values)

        if self.debug_path is not None:
            pdf_debug = PdfPages(self.debug_path+tags['model_name']+'.pdf')

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(x_values > l, x_values < h))
            filtered_y_values = y_values[filter].astype(np.float)
            filtered_weights = weights[filter].astype(float)
            # filtered_y_values = filtered_y_values * filtered_weights

            range_ = np.min(filtered_y_values), np.max(filtered_y_values)
            y, bins = np.histogram(filtered_y_values,
                                   range=range_,
                                   bins=int(np.sqrt(len(filtered_y_values))),
                                   weights=filtered_weights,
                                   density=False)
            bins_o = bins
            bins = (bins[1:] + bins[0:-1]) / 2


            fitted, cov = curve_fit(gauss_fnc, bins, y,
                                    p0=(np.mean(filtered_y_values), np.max(y), np.std(filtered_y_values)),
                                    sigma=np.sqrt(y),
                                    absolute_sigma=True
                                    )
            e = np.sqrt(np.diag(cov))
            e = e[2]
            mu, a, sigma = fitted

            print("XYZ 2", range_,l,h, mu, len(filtered_y_values), np.min(x_values[filter]), np.max(x_values[filter]))

            if self.debug_path is not None:
                fig, ax = plt.subplots()
                y = y.tolist()
                ax.step(bins_o, [y[0]] + y)


                xx = np.linspace(range_[0], range_[1], num=400)
                ax.plot(xx, gauss_fnc(xx, mu, a, sigma), color='r', ls='--',
                        label='[%d - %d) %s'%(int(l), int(h), tags['model_name']))
                ax.legend()
                pdf_debug.savefig(fig)
                plt.close(fig)


            # m = np.mean(filtered_y_values)
            # relvar = (filtered_y_values - m) / m
            # rms = np.sqrt(np.sum(filtered_weights * relvar ** 2) / np.sum(filtered_weights))
            # m = mu
            rms = sigma / mu

            # m = (np.std(filtered_y_values - m) / m)
            mean.append(rms)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(e)

        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        if self.debug_path is not None:
            pdf_debug.close()

        return processed_data

class ResolutionPlot(General2dBinningPlot):
    def __init__(self,**kwargs):
        super(ResolutionPlot, self).__init__(**kwargs)

    def draw(self, name_tag_formatter=None, return_fig=False, ax=None, **kwargs):
        fig = super().draw(name_tag_formatter, return_fig=True, ax=ax, **kwargs)
        axes = fig.axes[0] if ax is None else ax
        # axes[0].axhline(1, 0, 1, ls='--', linewidth=0.5, color='gray')
        axes.axhline(0, 0, 1, ls='--', linewidth=0.5, color='gray')
        if return_fig:
            return fig

    def _compute(self, x_values, y_values, weights=None):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []

        lows = []
        highs = []
        error = []

        bug_in_error_comptuation=True

        if weights is None:
            bug_in_error_comptuation=False
            weights = np.ones_like(y_values)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(x_values > l, x_values < h))
            filtered_y_values = y_values[filter].astype(np.float)
            filtered_weights = weights[filter].astype(float)
            # filtered_y_values = filtered_y_values * filtered_weights

            m = np.mean(filtered_y_values)
            relvar = (filtered_y_values - m) / m
            rms = np.sqrt(np.sum(filtered_weights * relvar ** 2) / np.sum(filtered_weights))

            # m = (np.std(filtered_y_values - m) / m)
            mean.append(rms)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(rms / np.sqrt(float(len(filtered_y_values))))



        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        return processed_data


class EfficiencyFakeRatePlot(General2dBinningPlot):
    def __init__(self,**kwargs):
        super(EfficiencyFakeRatePlot, self).__init__(**kwargs)

    def _compute(self, x_values, y_values, weights=None):
        e_bins = self.e_bins
        e_bins_n = np.array(e_bins)
        e_bins_n = (e_bins_n - e_bins_n.min()) / (e_bins_n.max() - e_bins_n.min())

        centers = []
        mean = []
        error = []

        lows = []
        highs = []

        bug_in_error_comptuation = True

        if weights is None:
            bug_in_error_comptuation = False
            weights = np.ones_like(y_values)

        for i in range(len(e_bins) - 1):
            l = e_bins[i]
            h = e_bins[i + 1]

            filter = np.argwhere(np.logical_and(x_values >= l, x_values < h))
            filtered_y_values = y_values[filter].astype(float)
            filtered_weights = weights[filter].astype(float)
            m = np.sum(filtered_y_values*filtered_weights)/np.sum(filtered_weights)
            mean.append(m)
            # print(np.sum(filtered_found), len(filtered_found), m, l, h)
            lows.append(l)
            highs.append(h)
            error.append(np.sqrt(m * (1 - m) / len(filtered_y_values)))

        hist_values, _ = np.histogram(x_values, bins=e_bins)
        # hist_values = (hist_values / (e_bins_n[1:] - e_bins_n[:-1])).tolist()
        # hist_values = (hist_values / np.sum(hist_values))

        processed_data = dict()
        processed_data['bin_lower_energy'] = np.array(lows)
        processed_data['bin_upper_energy'] = np.array(highs)
        processed_data['hist_values'] = hist_values
        processed_data['mean'] = np.array(mean)
        if not bug_in_error_comptuation:
            processed_data['error'] = np.array(error)

        return processed_data



class ResponseAndResolutionCombinedPlot():
    def __init__(self, y_label1='Response', y_label2='Resolution',resolution_by_fit=False,debug_path=None,ylim2=None,**kwargs):

        # kwargs2 = kwargs.copy()
        # kwargs2['y_label']='Response'
        # kwargs['y_label']='Resolution'
        self.response_plot = (ResponsePlot if not resolution_by_fit else ResponseByFitPlot)(ylim=[0.1, 2], y_label=y_label1, **kwargs)
        if resolution_by_fit:
            self.resolution_plot = (ResolutionPlot if not resolution_by_fit else ResolutionByFitPlot)(ylim=ylim2, y_label=y_label2,debug_path=debug_path,**kwargs)
        else:
            self.resolution_plot = (ResolutionPlot if not resolution_by_fit else ResolutionByFitPlot)(ylim=ylim2, y_label=y_label2,**kwargs)

    # def _compute(self, x_values, y_values, weights=None):
    #     self.response_plot._compute(x_values, y_values,weights)
    #     self.resolution_plot._compute(x_values, y_values,weights)

    def add_raw_values(self, x_values, y_values, tags={}, weights=None):
        self.response_plot.add_raw_values(x_values, y_values, tags=tags, weights=weights)
        self.resolution_plot.add_raw_values(x_values, y_values, tags=tags, weights=weights)

    def draw(self, name_tag_formatter=None, legend_loc_1=None, legend_loc_2=None, return_fig=True,return_ax=False,**kwargs):
        fig, [ax1,ax2] = plt.subplots(2, 1, figsize=(9, 6),gridspec_kw = {'wspace':0, 'hspace':0.05})

        self.response_plot.draw(name_tag_formatter, ax=ax1,legend_loc=legend_loc_1, **kwargs)
        self.resolution_plot.draw(name_tag_formatter, ax=ax2,legend_loc=legend_loc_2,**kwargs)
        ax1.xaxis.set_ticklabels([])
        ax1.set_xlabel(None)

        ax1.legend(loc=legend_loc_1, ncol=2)
        ax2.legend(loc=legend_loc_2, ncol=2)

        if return_ax:
            return fig, ax1, ax2
        return fig


class EffFakeRateCombinedPlot():
    def __init__(self, y_label1='', y_label2='', **kwargs):

        self.eff_plot = EfficiencyFakeRatePlot(y_label = y_label1, **kwargs)
        self.fake_rate_plot = EfficiencyFakeRatePlot(y_label = y_label2, **kwargs)


    def add_raw_values_1(self, x_values, y_values, tags={}, weights=None):
        self.eff_plot.add_raw_values(x_values, y_values, tags=tags, weights=weights)

    def add_raw_values_2(self, x_values, y_values, tags={}, weights=None):
        self.fake_rate_plot.add_raw_values(x_values, y_values, tags=tags, weights=weights)

    def draw(self, name_tag_formatter=None, return_ax=False, **kwargs):
        fig, [ax1,ax2] = plt.subplots(2, 1, figsize=(9, 6),gridspec_kw = {'wspace':0, 'hspace':0.05})
        self.eff_plot.draw(name_tag_formatter, ax=ax1, **kwargs)
        self.fake_rate_plot.draw(name_tag_formatter, ax=ax2, **kwargs)
        ax1.xaxis.set_ticklabels([])
        ax1.set_xlabel(None)
        if return_ax:
            return fig, ax1, ax2
        return fig
