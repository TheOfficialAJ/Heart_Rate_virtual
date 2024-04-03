import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from transforms import butter_bandpass_filter
from scipy.signal import find_peaks


# import line_profiler
class HeartRate:
    def __init__(self):
        self.segmented_img_plot = None
        self.heartrate_plot = None
        self.signal_plot = None
        self.signal = None
        self.hr_emas = []
        self.timestamp = []
        self.motion = None
        self.subplots = None
        self.heartrate = []
        self.mean_colors = {'nose': [], 'forehead': [], 'cheeks': []}
        self.butter_params = {'_lowcut': 0.5, '_highcut': 2.5, '_fs': 30, 'order': 3}
        self.method = None

    def set_signal(self, signal):
        self.signal = signal

    def estimateHR_FFT(self, smooth_signal, thresh=0.0, get_power_spectrum=False):
        freq_min = 0.8
        freq_max = 2
        freqs = scipy.fftpack.fftfreq(smooth_signal.shape[0], d=1.0 / 30)
        fft = scipy.fftpack.fft(smooth_signal)
        fft_maximums = []
        for i in range(fft.shape[0]):
            if freq_min <= freqs[i] <= freq_max:
                fftMap = abs(fft[i])
                fft_maximums.append(fftMap.max())
            else:
                fft_maximums.append(0)
        if get_power_spectrum:
            return freqs
        peaks, properties = find_peaks(fft_maximums)
        max_peak = -1
        max_freq = 0

        # Find frequency with max amplitude in peaks
        for peak in peaks:
            if fft_maximums[peak] > max_freq:
                max_freq = fft_maximums[peak]
                max_peak = peak
        if max_peak == -1:
            print("Dead haha")
            return 0
        return freqs[max_peak] * 60

    def recalculate_limits(self):
        for axes in self.subplots[1]:
            axes.relim()
            axes.autoscale_view(True, True, True)

    def initPlot(self):
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        self.subplots = fig, [ax1, ax2, ax3]
        self.signal_plot, = self.subplots[1][0].plot([], [], 'b-')
        self.heartrate_plot, = self.subplots[1][1].plot([], [], 'r-')
        self.segmented_img_plot = self.subplots[1][2].imshow(np.zeros((100, 100, 3)))
        self.subplots[1][0].set_title(f'Signal {self.method}')
        self.subplots[1][1].set_title('Heart Rate')
        self.subplots[1][0].set_xlabel('Time (s)')
        for axes in self.subplots[1]:
            axes.set_autoscale_on(True)  # enable autoscale
            axes.autoscale_view(True, True, True)

    def plotHR(self):
        if self.subplots is None:
            self.initPlot()

        timestamp = np.linspace(0, len(self.signal) / 30, len(self.signal))
        print(len(self.signal), len(timestamp))
        self.signal_plot.set_ydata(self.signal)
        self.signal_plot.set_xdata(timestamp)

        self.heartrate_plot.set_xdata(np.linspace(0, len(self.signal)/30, len(self.heartrate)))
        self.heartrate_plot.set_ydata(self.heartrate)
        self.recalculate_limits()
        self.subplots[0].canvas.draw()
        self.subplots[0].canvas.flush_events()

    def getCHROMfromBGR(self, colors):
        X = 3 * colors[:, 2] - 2 * colors[:, 0]
        Y = 1.5 * colors[:, 2] + colors[:, 1] - 1.5 * colors[:, 0]
        print(f"x shape: {np.shape(X)}")
        alpha = np.std(X, axis=0) / np.std(Y, axis=0)
        result = X - alpha * Y
        return result

    def LGIfromPyVHR(self, B, G, R):
        # print(B.shape)
        X = np.zeros((1, 3, B.shape[0]))
        X[:, 0, :] = R
        X[:, 1, :] = G
        X[:, 2, :] = B
        print(f'sape of X {X.shape}')
        U, _, _ = np.linalg.svd(X)
        print(f'shape of U {U.shape}')
        S = U[:, :, 0]
        S = np.expand_dims(S, 2)
        print(f'shape of S {S.shape}')
        sst = np.matmul(S, np.swapaxes(S, 1, 2))
        print(f'shape of sst {sst.shape}')
        p = np.tile(np.identity(3), (S.shape[0], 1, 1))
        print(f'shape of p {p.shape}')
        P = p - sst
        print(p.shape)
        print(X.shape)
        Y = np.matmul(P, X)
        bvp = Y[:, 1, :]
        # bvp = np.expand_dims(bvp,axis=1)
        print(bvp.shape)
        print(f"bvp before transform {bvp}")
        # Normalise bvp
        bvp = bvp - np.mean(bvp)
        bvp = bvp / np.std(bvp)
        bvp = bvp.reshape((-1, 1)).squeeze()
        return bvp

    def POSfromPyVHR(self, B, G, R, **kargs):

        eps = 10 ** -9
        X = np.zeros((1, 3, B.shape[0]))
        X[:, 0, :] = R
        X[:, 1, :] = G
        X[:, 2, :] = B
        e, c, f = X.shape  # e = #estimators, c = 3 rgb ch., f = #frames
        w = int(1.6 * kargs['fps'])  # window length

        # stack e times fixed mat P
        P = np.array([[0, 1, -1], [-2, 1, 1]])
        Q = np.stack([P for _ in range(e)], axis=0)

        # Initialize (1)
        H = np.zeros((e, f))
        for n in np.arange(w, f):
            # Start index of sliding window (4)
            m = n - w + 1
            # Temporal normalization (5)
            Cn = X[:, :, m:(n + 1)]
            M = 1.0 / (np.mean(Cn, axis=2) + eps)
            M = np.expand_dims(M, axis=2)  # shape [e, c, w]
            Cn = np.multiply(M, Cn)

            # Projection (6)
            S = np.dot(Q, Cn)
            S = S[0, :, :, :]
            S = np.swapaxes(S, 0, 1)  # remove 3-th dim

            # Tuning (7)
            S1 = S[:, 0, :]
            S2 = S[:, 1, :]
            alpha = np.std(S1, axis=1) / (eps + +np.std(S2, axis=1))
            alpha = np.expand_dims(alpha, axis=1)
            Hn = np.add(S1, alpha * S2)
            Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
            # Overlap-adding (8)
            H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)
        H = H.reshape((-1, 1))
        # Make H one dimensional
        H = H.squeeze()
        return H

    def PCAfromPyVHR(self, B, G, R, **kargs):
        bvp = []
        signal = np.zeros([1, 3, B.shape])
        signal[:, 0, :] = R
        signal[:, 1, :] = G
        signal[:, 2, :] = B
        for i in range(signal.shape[0]):
            X = signal[i]
            pca = PCA(n_components=3)
            pca.fit(X)

            # selector
            if kargs['component'] == 'all_comp':
                bvp.append(pca.components_[0] * pca.explained_variance_[0])
                bvp.append(pca.components_[1] * pca.explained_variance_[1])
            elif kargs['component'] == 'second_comp':
                bvp.append(pca.components_[1] * pca.explained_variance_[1])
        bvp = np.array(bvp)

        return bvp

    def set_method(self, method):
        self.method = method

    # @profile
    def plot_signal(self, signal, label, axis=None):
        self.timestamp = np.linspace(0, len(signal) / 30, len(signal))
        if axis is None:
            plt.plot(self.timestamp, signal, label=label)
        else:
            axis.plot(self.timestamp, signal, label=label)
        plt.legend()
        plt.draw()
        plt.pause(0.001)
        plt.clf()

    def update_image(self, region_extractor):
        if self.subplots is None:
            self.initPlot()
        image = region_extractor.images[-1]
        mask = region_extractor.masks[-1][0]
        image = cv2.bitwise_and(image, image, mask=mask)
        self.segmented_img_plot.set_data(image)
        self.recalculate_limits()
        self.subplots[0].canvas.draw()
        self.subplots[0].canvas.flush_events()


    # @profile
    def regions_to_signal(self, masks, images):
        nose_masks = masks[:, 0]
        print(masks[:, 0].shape, images.shape)
        mean_colors = [self.mean_color_mask(image, mask) for mask, image in zip(nose_masks, images)]
        # Normalise the mean colors
        mean_colors = np.array(mean_colors)
        mean_colors = mean_colors / 255
        mean_colors = mean_colors.squeeze()
        mean_colors = mean_colors.squeeze()
        self.mean_colors['nose'].extend(mean_colors)
        print(mean_colors)

    def process_signal(self, region_extractor):
        # Find the detrending intervals from motion
        self.motion = region_extractor.motions
        motion_peaks, _ = find_peaks(self.motion, height=0.5)
        self.breakpoints = motion_peaks
        masks = region_extractor.masks
        images = region_extractor.images
        self.regions_to_signal(masks, images)
        if self.method == 'FFT':
            self.set_signal(self.mean_colors['nose'])
        elif self.method == "CHROM":
            self.set_signal(self.getCHROMfromBGR(np.array(self.mean_colors['nose'])))
        elif self.method == 'POS':
            self.set_signal(self.POSfromPyVHR(*np.array(self.mean_colors['nose']).transpose(1, 0), fps=30))
        elif self.method == 'LGI':
            self.set_signal(self.LGIfromPyVHR(*np.array(self.mean_colors['nose']).transpose(1, 0)))
        # Detrend and smoothen the signal
        self.signal = scipy.signal.detrend(self.signal, bp=self.breakpoints)
        self.signal = butter_bandpass_filter(self.signal, self.butter_params['_lowcut'], self.butter_params['_highcut'],
                                             self.butter_params['_fs'], self.butter_params['order'])

    def estimateHR(self, plot=True):
        self.heartrate.append(self.estimateHR_FFT(self.signal))
        # Take exponential moving average of last 9 values
        # heartrate = 0
        # for i in range(1, 10):
        #     heartrate += self.heartrate[-i]*

        if plot:
            self.plotHR()

    # @profile
    def mean_color_mask(self, image, mask):
        mean_color = cv2.mean(image, mask=mask)
        return mean_color[:3]
