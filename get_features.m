features = zeros(size(x,1), 128);
impatch_or = zeros(feature_width,feature_width,size(x,1));
image_filtered = zeros(size(image));
imgrad_mag = zeros(size(image));
imgrad_orient = zeros(size(image));
imghist = zeros([size(x,1), (feature_width/4)^2 * 8 ]);

% Create Gaussian filter
Sigma = 0.5;
Hsize = 3; 
H = fspecial('gaussian', Hsize,Sigma);

% Filter image with gaussian, and take gradient 
image_filtered = imfilter (image, H,'symmetric','same','conv');
[imgrad_mag , imgrad_orient] = imgradient(image_filtered, 'sobel');

% Create blockproc function for histcounts
fun = @(block_struct) histcounts(block_struct.data, [-180:45:180]);

%Create 16x16 image patches around interest points
for i = 1:size(x,1)   
    impatch_or(:,:,i) = imgrad_orient((y(i)-7):(y(i)+8),(x(i)-7):(x(i)+8));    
    hist_temp = blockproc(impatch_or(:,:,i),[feature_width/4 feature_width/4], fun);
    imghist(i,:) = reshape(hist_temp',[1 (8*(feature_width/4)^2)]);
    
    imghist(i,:) = imghist(i,:)/norm(imghist(i,:),2);
    tempy = find(imghist(i,:) > 0.2);
    imghist(i,tempy) = 0.2;
    imghist(i,:) = imghist(i,:)/norm(imghist(i,:),2);
end