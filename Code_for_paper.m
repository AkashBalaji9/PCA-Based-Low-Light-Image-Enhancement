[ent_out, ssimval, FSIM_c, FSIMc_c, mpcqi, vsi, niqe_out] = deal(ones(10,6));
%% 
%Image Number
k = 3;
%Method 1 : PCA based Low Light Image Enhancement using Reflection Model
%Reading image as rgb          
img_rgb = imread('test3.png');
rgb_s = double(img_rgb);

%max and min values of each channel
max_r  = max(rgb_s(:,:,1),[],'all');               
min_r  = min(rgb_s(:,:,1),[],'all');
max_g  = max(rgb_s(:,:,2),[],'all');
min_g  = min(rgb_s(:,:,2),[],'all');
max_b  = max(rgb_s(:,:,3),[],'all');
min_b  = min(rgb_s(:,:,3),[],'all');

%Step 1 : Channel Stretching
rgb_s(:,:,1) = (rgb_s(:,:,1)- min_r)/(max_r - min_r);
rgb_s(:,:,2) = (rgb_s(:,:,2)- min_g)/(max_g - min_g);
rgb_s(:,:,3) = (rgb_s(:,:,3)- min_b)/(max_b - min_b);

%Step 2 : Joining stretched channels and RGB to HSV                
HSV = rgb2hsv(img_rgb);                
[h,s,v] = imsplit(HSV);
HSV_s = rgb2hsv(rgb_s);
[h_s,s_s,v_s] = imsplit(HSV_s);

%Step 3 : Estimation of reflection coeficient
%Application of 2-D Gaussian filter with scale = 15,80,250 and weight = 1/3
Iv_illu = (1/3)*imgaussfilt(v_s,15) + (1/3)*imgaussfilt(v_s,80) + (1/3)*imgaussfilt(v_s,250); 

%Step 4 : Adaptive generation of two enhanced images for PCA fusion
Iv_en1 = v_s;
Iv_en2 = v_s;
mean_sat = mean2(s_s);
K1 = 0.3*mean_sat;
K2 = 0.8*mean_sat;
for i4 = 1:size(v_s,1)
    for j4 = 1:size(v_s,2)
        max_val = max(v_s(i4,j4),Iv_illu(i4,j4));
        Iv_en1(i4,j4) = (v_s(i4,j4)*(1+K1))/(max_val+K1);
        Iv_en2(i4,j4) = (v_s(i4,j4)*(1+K2))/(max_val+K2);
    end
end

%Step 5 : PCA Fusion of enhanced images
C = cov([Iv_en1(:) Iv_en2(:)]);
[V, D] = eig(C);
if D(1,1) >= D(2,2)
  pca = V(:,1)./sum(V(:,1));
else  
  pca = V(:,2)./sum(V(:,2));
end
Fus = pca(1)*Iv_en1 + pca(2)*Iv_en2;

%Step 6 : Contrast Limited Adaptive Histogram Equalization
J = adapthisteq(Fus);
J1 = uint8(J*255);
hsv_new = cat(3,h_s,s_s,J);
out_rgb = hsv2rgb(hsv_new);
out_rgb = uint8(out_rgb*255);

%Method 2 : HE (Histogram Equalization)
v_he = histeq(v);
hsv_he = cat(3,h,s,v_he);
out_rgb2 = hsv2rgb(hsv_he);
out_rgb2 = uint8(out_rgb2*255);

%Method 3 : CLAHE (Contrast Limited Adaptive Histogram Equalization)
v_clahe = adapthisteq(v);
hsv_clahe = cat(3,h,s,v_clahe);
out_rgb3 = hsv2rgb(hsv_clahe);
out_rgb3 = uint8(out_rgb3*255);

%Method 4 : DCP (De-Hazing using Dark Channel Prior)
im_double = double(img_rgb);
im_double = im_double./255;
out_rgb4 = deHaze(im_double);
out_rgb4 = uint8(out_rgb4*255);

%Method 5 : FEA (Fast Efficient Algorithm for low lighting enhancement)
iminv = imcomplement(img_rgb);
Binv = imreducehaze(iminv, 'Method', 'approx', 'ContrastEnhancement', 'boost');
Bimp = imcomplement(Binv);
%out_rgb5 = imguidedfilter(Bimp);
out_rgb5 = Bimp;

%Method 6 : MSRCR (Multi Scale Retinex with Colour Restoration)
scales = [2 120 240];
alpha = 500;
d = 1.5;
out_rgb6 = MSRCR(im_double,scales,[],alpha,d);
out_rgb6 = uint8(out_rgb6*255);

%Method 7 : JHE (Joint Histogram Equalization)
w = 3;
kernel = ones(w) / w^2;
avgimg = imfilter(img_rgb,kernel);

%Displaying Image Results
figure
subplot(2,4,1)
imagesc(img_rgb)
title 'Original'
axis image off
subplot(2,4,2)
imagesc(out_rgb)
title 'Proposed'
axis image off
subplot(2,4,3)
imagesc(out_rgb2)
title 'HE'
axis image off
subplot(2,4,4)
imagesc(out_rgb3)
title 'CLAHE'
axis image off
subplot(2,4,5)
imagesc(out_rgb4)
title 'DCP'
axis image off
subplot(2,4,6)
imagesc(out_rgb5)
title 'FEA'
axis image off
subplot(2,4,7)
imagesc(out_rgb6)
title 'MSRCR'
axis image off

%Objective Analysis
out_arr = {out_rgb, out_rgb2, out_rgb3, out_rgb4, out_rgb5, out_rgb6};
ent_in = entropy(img_rgb);
niqe_in = niqe(img_rgb);

for i5 = 1:6
    ent_out(k,i5) = entropy(out_arr{1,i5});
    ssimval(k,i5) = ssim(out_arr{1,i5},img_rgb);
    [FSIM_c(k,i5),FSIMc_c(k,i5)] = FeatureSIM(out_arr{1,i5},img_rgb);
    mpcqi(k,i5) = PCQI(rgb2gray(out_arr{1,i5}),rgb2gray(img_rgb));
    vsi(k,i5) = VSI(out_arr{1,i5},img_rgb);
    niqe_out(k,i5) = niqe(out_arr{1,i5});
end
%% 
figure
subplot(2,4,1)
boxplot(ent_out);
title 'Entropy'
subplot(2,4,2)
boxplot(ssimval);
title 'SSIM'
subplot(2,4,3)
boxplot(FSIM_c);
title 'FSIM'
subplot(2,4,4)
boxplot(FSIMc_c);
title 'FSIMc'
subplot(2,4,5)
boxplot(mpcqi);
title 'PCQI'
subplot(2,4,6)
boxplot(vsi);
title 'VSI'
subplot(2,4,7)
boxplot(niqe_out);
title 'NIQE'
subplot(2,4,8)
legend = imread('legend.png');
imagesc(legend);
axis image off
