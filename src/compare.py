from haarPsi import haar_psi
import mpl_interaction as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
import sys
import os.path
import imghdr
import time
import cv2
import numpy as np
import argparse
import psutil
import math
from skimage.measure import compare_ssim
import imutils
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

process = psutil.Process(os.getpid())
FILE_SIZE_THRESHOLD = 524288  # 512KB
OUTPUTNAME = "output"


def now(): return int(round(time.time() * 1000))


def convert_size(size_bytes):
   if size_bytes == 0:
	   return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def parseArgs(argv):
	parser = argparse.ArgumentParser(description='Compare two images')

	parser.add_argument('-l', '--left',
                     type=str,
                     required=True,
                     help='Specify left image to be compared',
                     dest='left')

	parser.add_argument('-r', '--right',
                     type=str,
                     required=True,
                     help='Specify right image to be compared',
                     dest='right')

	parser.add_argument('--verbose',
                     required=False,
                     help='Enable/Disable log information',
                     action='store_true')

	subparsers = parser.add_subparsers(dest='type')

	colour_mode = subparsers.add_parser('colour')

	colour_mode.add_argument('-m', '--mode',
                          type=str,
                          required=False,
                          choices=['rgb', 'rgba', 'a'],
                          default='rgba',
                          help='Specify the colour comparison mode',
                          dest='mode')

	colour_mode.add_argument('--logdiff',
                          required=False,
                          help='Execute logarithmic comparison',
                          action='store_true')

	colour_mode.add_argument('--fillalpha',
                          required=False,
                          help='Add alpha channel to an RGB image',
                          action='store_true')

	subparsers.add_parser('ssim')

	subparsers.add_parser('haarpsi')

	brisk_mode = subparsers.add_parser('brisk')

	brisk_mode.add_argument('--mismatched',
                         required=False,
                         help='Display mismatched keypoints location',
                         action='store_true')

	return parser.parse_args(args=argv)


def compare(argv):
	args = parseArgs(argv)
	filename1 = args.left
	filename2 = args.right
	log = args.verbose
	try:
		if not os.path.exists(filename1) or not os.path.isfile(filename1) or not \
			os.path.exists(filename2) or not os.path.isfile(filename2):
			sys.stderr.write('Impossible to locate input files\n')
			sys.exit(-2)

		if imghdr.what(filename1) is None or imghdr.what(filename2) is None:
			sys.stderr.write('Unrecognized image format\n')
			sys.exit(-3)

		bigFiles = (os.path.getsize(filename1) >
                    FILE_SIZE_THRESHOLD or os.path.getsize(filename2) > FILE_SIZE_THRESHOLD)
		start = 0
		if log:
			start = now()
			sys.stdout.write('{:21}'.format('Opening images... '))
		if not bigFiles:
			im1 = cv2.imread(filename1, cv2.IMREAD_UNCHANGED)
			im2 = cv2.imread(filename2, cv2.IMREAD_UNCHANGED)
		else:
			pool = ThreadPool(processes=1)
			res = pool.apply_async(cv2.imread, (filename1, cv2.IMREAD_UNCHANGED,))
			im2 = cv2.imread(filename2, cv2.IMREAD_UNCHANGED)
			im1 = res.get()
		if log:
			sys.stdout.write('Done!\n\n')

		if im1.size == 0 or im2.size == 0:
			sys.stderr.write(
				filename1 + ' cannot be read\n') if im1.size == 0 else sys.stderr.write(filename2 + ' cannot be read\n')
			sys.exit(-4)

		size1 = im1.shape
		size2 = im2.shape
		if size1[2] < 3 or size2[2] < 3:
			sys.stderr.write(filename1 + ' has less than 3 colour channels\n') if size1[2] < 3 else sys.stderr.write(
				filename2 + ' has less than 3 colour channels\n')
			sys.exit(-5)
		if size1[0] != size2[0] or size1[1] != size2[1]:
			sys.stderr.write(
				'Impossible to compare images: the sizes don\'t match\n')
			sys.exit(-6)

		numberofpixels = size1[0] * size1[1]
		if log:
			print('{:21}'.format('Left image path: ') + os.path.abspath(filename1))
			print('{:21}'.format('Left image channel: ') +
                            ('RGB' if size1[2] == 3 else 'RGBA'))
			print('{:21}'.format('Right image path: ') + os.path.abspath(filename2))
			print('{:21}'.format('Right image channel: ') +
                            ('RGB' if size2[2] == 3 else 'RGBA'))
			print('{:21}'.format('Width: ') + str(size1[1]))
			print('{:21}'.format('Height: ') + str(size1[0]))
			print('{:21}'.format('Number of pixels: ') + str(numberofpixels))
			print('\n' + '-'*40 + '\n')

		if 'ssim' in args.type:
			if log:
				print('Executing SSIM\n')
			grayA = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
			(score, diff) = compare_ssim(grayA, grayB, full=True)
			diff = (diff * 255).astype("uint8")
			if log:
				print('{:21}'.format('SSIM: ') + '{:10}'.format(str(score)))
			out = cv2.threshold(
                            diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		if 'haarpsi' in args.type:
			if log:
				print('Executing HaarPSI\n')
			if size1[2] == 4:
				im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)
			if size2[2] == 4:
				im2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2BGR)
			(score, _, out) = haar_psi(im1, im2, False)
			if log:
				print('{:21}'.format('HaarPSI: ') + '{:10}'.format(str(score)))
			out = cv2.cvtColor(out.astype('uint8'), cv2.COLOR_BGR2BGRA)
			out[:, :, 3] = 255
		elif 'brisk' in args.type:
			if log:
				print('Executing BRISK\n')
			brisk = cv2.BRISK_create(thresh=10, octaves=1)
			if size1[2] == 4:
				im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)
			if size2[2] == 4:
				im2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2BGR)
			grayA = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
			grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
			kp1, des1 = brisk.detectAndCompute(grayA, None)
			kp2, des2 = brisk.detectAndCompute(grayB, None)
			matcher = cv2.BFMatcher(cv2.NORM_L2SQR)
			matches = matcher.match(des1, des2)
			distances = [match.distance for match in matches]
			min_dist = min(distances)
			avg_dist = sum(distances) / len(distances)
			min_multiplier_tolerance = 10
			min_dist = min_dist or avg_dist * 1.0 / min_multiplier_tolerance
			good = [match for match in matches if
                            match.distance <= min_multiplier_tolerance * min_dist]

			if not args.mismatched:
				src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
				dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
				matchesMask = mask.ravel().tolist()

				h, w, _ = im1.shape
				pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                     ).reshape(-1, 1, 2)
				dst = cv2.perspectiveTransform(pts, M)

				im2 = cv2.polylines(im2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

				draw_params = dict(matchColor=(0, 255, 0),
                                    singlePointColor=(255, 255, 255),
                                    matchesMask=matchesMask,
                                    flags=2)

				out = cv2.drawMatches(im1, kp1, im2, kp2, good, None, **draw_params)
			else:
				kp1_matched = ([kp1[m.queryIdx] for m in good])
				kp2_matched = ([kp2[m.trainIdx] for m in good])
				kp1_miss_matched = [kp for kp in kp1 if kp not in kp1_matched]
				kp2_miss_matched = [kp for kp in kp2 if kp not in kp2_matched]
				out_1 = cv2.drawKeypoints(im1, kp1_miss_matched, None,
                                    color=(0, 255, 255), flags=0)
				out_2 = cv2.drawKeypoints(im2, kp2_miss_matched, None,
                                    color=(255, 255, 192), flags=0)
				out = np.concatenate((out_1, out_2), axis=1)
			out = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)
			out[:, :, 3] = 255

			if log:
				print('{:21}'.format('Total matches: ') +
                                    str(len(good)) + '/' + str(len(matches)))
		elif 'colour' in args.type:
			mode = args.mode
			fillalpha = args.fillalpha
			if mode == 'rgb':
				if size1[2] == 4:
					im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)
				if size2[2] == 4:
					im2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2BGR)
			elif mode == 'rgba':
				if size1[2] == 4 and size2[2] == 4:
					pass
				elif not fillalpha and size1[2] == 3 and size2[2] == 3:
					mode = 'rgb'
				elif not fillalpha and (size1[2] == 4 or size2[2] == 4):
					if size1[2] == 4:
						im1 = cv2.cvtColor(im1, cv2.COLOR_BGRA2BGR)
					if size2[2] == 4:
						im2 = cv2.cvtColor(im2, cv2.COLOR_BGRA2BGR)
					mode = 'rgb'
				else:
					if size1[2] == 3 and fillalpha:
						im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2BGRA)
					if size2[2] == 3 and fillalpha:
						im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2BGRA)
			elif mode == 'a':
				if size1[2] == 3 or size2[2] == 3:
					sys.stderr.write('Impossible to compare images using the specified comparison mode: ' +
                                            (filename1 if size1[2] == 3 else filename2) + ' doesn\'t have an alpha channel\n')
					sys.exit(-7)
				im1 = im1[:, :, 3]
				im2 = im2[:, :, 3]

			if log:
				print('{:21}'.format('Comparison mode: ') + mode.upper())
				sys.stdout.write('{:21}'.format('Computing diff... '))

			im1_im2 = cv2.subtract(im1, im2)
			im2_im1 = cv2.subtract(im2, im1)
			diff = cv2.max(im1_im2, im2_im1)
			im1_im2 = None
			im2_im1 = None
			if mode != 'a':
				maxDiff = np.amax(diff, axis=2)
				diff = None
			else:
				maxDiff = diff

			noDiff = 0
			precDiff = 0
			smallDiff = 0
			mediumDiff = 0
			largeDiff = 0
			out = np.zeros((size1[0], size1[1], 4), dtype='uint8')
			if not args.logdiff:
				indeces = np.argwhere(maxDiff == 0)
				noDiff = indeces.shape[0]
				if noDiff == numberofpixels:
					sys.stdout.write('Images identical!\n')
				else:
					sys.stdout.write(
                                            'Images Different!\n\n' if log else 'Images Different!\n')
					if not bigFiles:
						indeces = np.argwhere(maxDiff == 1)
						precDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [255, 0, 0, 0]
						indeces = None
						i = None
						indeces = np.argwhere(maxDiff == 2)
						smallDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [0, 255, 255, 0]
						indeces = None
						i = None
						indeces = np.logical_or(maxDiff == 3, maxDiff == 4).nonzero()
						mediumDiff = indeces[0].shape[0]
						out[indeces] = [0, 165, 255, 0]
						indeces = None
						indeces = np.argwhere(maxDiff > 4)
						largeDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [0, 0, 255, 0]
						indeces = None
						i = None
					else:
						b = pool.apply_async(np.argwhere, (maxDiff == 1,))
						y = pool.apply_async(np.argwhere, (maxDiff == 2,))
						o = pool.apply_async(np.logical_or, (maxDiff == 3, maxDiff == 4,))
						indeces = np.argwhere(maxDiff > 4)
						largeDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [0, 0, 255, 0]
						indeces = None
						i = None
						indeces = b.get()
						precDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [255, 0, 0, 0]
						indeces = None
						i = None
						indeces = y.get()
						smallDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [0, 255, 255, 0]
						indeces = None
						i = None
						indeces = o.get().nonzero()
						mediumDiff = indeces[0].shape[0]
						out[indeces] = [0, 165, 255, 0]
						indeces = None
			else:
				indeces = np.argwhere(maxDiff < 2)
				noDiff = indeces.shape[0]
				if noDiff == numberofpixels:
					sys.stdout.write('Images identical!\n')
				else:
					sys.stdout.write(
                                            'Images Different!\n\n' if log else 'Images Different!\n')
					if not bigFiles:
						indeces = np.logical_or(maxDiff == 2, maxDiff == 3).nonzero()
						precDiff = indeces[0].shape[0]
						out[indeces] = [255, 0, 0, 0]
						indeces = None
						indeces = np.logical_and(maxDiff >= 4, maxDiff < 8).nonzero()
						smallDiff = indeces[0].shape[0]
						out[indeces] = [0, 255, 255, 0]
						indeces = None
						indeces = np.logical_and(maxDiff >= 8, maxDiff < 16).nonzero()
						mediumDiff = indeces[0].shape[0]
						out[indeces] = [0, 165, 255, 0]
						indeces = None
						indeces = np.argwhere(maxDiff >= 16)
						largeDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [0, 0, 255, 0]
						indeces = None
						i = None
					else:
						b = pool.apply_async(np.logical_or, (maxDiff == 2, maxDiff == 3,))
						y = pool.apply_async(np.logical_and, (maxDiff >= 4, maxDiff < 8,))
						o = pool.apply_async(np.logical_and, (maxDiff >= 8, maxDiff < 16,))
						indeces = np.argwhere(maxDiff >= 16)
						largeDiff = indeces.shape[0]
						i = (np.array(indeces[:, 0]), np.array(indeces[:, 1]))
						out[i] = [0, 0, 255, 0]
						indeces = None
						i = None
						indeces = b.get().nonzero()
						precDiff = indeces[0].shape[0]
						out[indeces] = [255, 0, 0, 0]
						indeces = None
						indeces = y.get().nonzero()
						smallDiff = indeces[0].shape[0]
						out[indeces] = [0, 255, 255, 0]
						indeces = None
						indeces = o.get().nonzero()
						mediumDiff = indeces[0].shape[0]
						out[indeces] = [0, 165, 255, 0]
						indeces = None
			# Set alpha to white
			out[:, :, 3] = 255

			if log and noDiff != numberofpixels:
				print('{:21}'.format('No Diff = ') + '{:10}'.format(str(noDiff)) +
                                    ' (' + '{0:.9f}'.format(
                                    (float(noDiff) / numberofpixels) * 100) + '%)')
				print('{:21}'.format('Precision Diff = ') + '{:10}'.format(str(precDiff)) +
                                    ' (' + '{0:.9f}'.format(
                                    (float(precDiff) / numberofpixels) * 100) + '%)')
				print('{:21}'.format('Small Diff = ') + '{:10}'.format(str(smallDiff)) +
                                    ' (' + '{0:.9f}'.format(
                                    (float(smallDiff) / numberofpixels) * 100) + '%)')
				print('{:21}'.format('Medium Diff = ') + '{:10}'.format(str(mediumDiff)) +
                                    ' (' + '{0:.9f}'.format(
                                    (float(mediumDiff) / numberofpixels) * 100) + '%)')
				print('{:21}'.format('Large Diff = ') + '{:10}'.format(str(largeDiff)) +
                                    ' (' + '{0:.9f}'.format(
                                    (float(largeDiff) / numberofpixels) * 100) + '%)\n')
		if log:
			print('\n' + '-'*40 + '\n')
			sys.stdout.write('{:21}'.format('Saving result... '))
		params = [cv2.IMWRITE_PNG_STRATEGY, 2, cv2.IMWRITE_PNG_COMPRESSION, 2]
		cv2.imwrite(OUTPUTNAME + '.png', out, params)
		if log:
			sys.stdout.write('Done!\n\n')

		if log:
			stop = now()
			time = stop - start
			sys.stdout.write('{:21}'.format('Total Time Taken = '))
			if time < 1000:
				print(str(time) + 'ms')
			elif time >= 1000 and time < 60000:
				print(str(time / 1000) + 's ' + str(time % 1000) + 'ms')
			else:
				m = time / 60000
				s = (time - m * 60000) / 1000
				ms = (time - m * 60000) % 1000
				print(str(m) + 'm ' + str(s) + 's ' + str(ms) + 'ms')
			print('{:21}'.format('Memory usage = ') +
                            convert_size(process.memory_info().rss))
		# Create 2x2 sub plots
		gs = gridspec.GridSpec(2, 2)

		fig = plt.figure()
		f = mpl.ZoomOnWheel(fig)
		ax1 = fig.add_subplot(gs[0, 0])  # row 0, col 0
		ax1.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
		           if size1[2] == 3 else cv2.cvtColor(im1, cv2.COLOR_BGRA2RGBA))
		plt.axis("off")

		ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)  # row 0, col 1
		ax2.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
		           if size1[2] == 3 else cv2.cvtColor(im2, cv2.COLOR_BGRA2RGBA))
		plt.axis("off")

		if 'brisk' not in args.type:
			# row 1, span all columns
			ax3 = fig.add_subplot(gs[1, :], sharex=ax1, sharey=ax1)
		else:
			ax3 = fig.add_subplot(gs[1, :])  # row 1, span all columns
		ax3.imshow(cv2.cvtColor(out, cv2.COLOR_BGRA2RGBA)
		           if not 'ssim' in args.type else out)
		plt.axis("off")

		mng = plt.get_current_fig_manager()
		mng.window.state('zoomed')
		plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0,
		                    top=1.0, hspace=0.02, wspace=0.0)
		plt.show()
	except Exception as e:
		sys.stderr.write('Error - ' + str(e) + ' (Line ' +
		                 str(sys.exc_info()[2].tb_lineno) + ')')
		sys.exit(-1)


if __name__ == "__main__":
	compare(sys.argv[1:])
