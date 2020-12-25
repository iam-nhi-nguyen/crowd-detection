function [im, seg1, seg2] = crowd(imOrig, nrows, ncols, dtheta, lambdaMin, sratio, nreps)
%CROWD Segment an image into crowd and non-crowd regions
%   [IM, SEG1, SEG2] = CROWD(IMORIG, NROWS, NCOLS, DTHETA, LAMBDAMIN,
%   SRATIO, NREPS) segments a color image IMORIG into 2 regions SEG1
%   and SEG2, one is a crowd region and the other is a non-crowd
%   region.

%   The function resizes the image IMORIG into a NROWS-by-NCOLS image
%   IM, and then converts it into grayscale. It then applys a set of
%   gabor filters with orientations evenly spaced from 0 (inclusively)
%   to 180 (exclusively) degrees with step DTHETA and wavelengths
%   log2-space evenly spaced from LAMBDAMIN (inclusively) to the radius
%   of the image (exclusively). With the resulting magnitudes for each
%   gabor filter, it applys a gaussian filter with standard deviation
%   SRATIO times the standard deviation of the corresponding gabor
%   filter. The results are flattened and normalized before being
%   segmented into 2 regions using kmean with NREPS replicates.

im = imresize(imOrig, [nrows ncols]);
imGray = rgb2gray(im);

% orientations (thetas)
thetas = 0:dtheta:(180 - dtheta);

% wavelengths (lambdas)
lambdaMax = hypot(nrows, ncols) / 2;        % radius of the image
n = floor(log2(lambdaMax / lambdaMin));
lambdas = 2 .^ (0:(n-1)) * lambdaMin;

% gabor filter bank
gabors = gabor(lambdas, thetas);
gabormags = imgaborfilt(imGray, gabors);

% feature extraction
features = zeros(size(gabormags));
for i = 1:length(gabors)
    gabormag = gabormags(:,:,i);
    sigma = 0.5 * gabors(i).Wavelength;
    features(:,:,i) = imgaussfilt(gabormag, sratio * sigma);
end

% spatial coordinations
X = 1:ncols;
Y = 1:nrows;
[X, Y] = meshgrid(X, Y);
features = cat(3, features, X);
features = cat(3, features, Y);

% flatten and normalize the filtered image
features = reshape(features, nrows * ncols, []);
features = (features - mean(features)) ./ std(features);

% segment the image into crowd and non-crowd
labels = kmeans(features, 2, 'Replicates', nreps);
labels = reshape(labels, [nrows ncols]);

% display the segmentations using the original image
seg1 = zeros(size(im), 'like', im);
seg2 = zeros(size(im), 'like', im);
mask = labels == 1;
mask = repmat(mask, [1 1 3]);
seg1(mask) = im(mask);
seg2(~mask) = im(~mask);

end