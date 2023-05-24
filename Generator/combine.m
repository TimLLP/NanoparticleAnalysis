% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/09/18
% Southwest university

clear,close all,clc;
% mask combine
maskdir1 = '/media/hadmatter/Adamberg/备份/data/test1/';
maskdir2 = '/media/hadmatter/Adamberg/备份/data/test2/';

result1 = stitching6(maskdir1);
result2 = stitching6(maskdir2);
result = [result1,result2];

%%
orimat = '/home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/Datasets/data.mat';
load(orimat);
clearvars -except result img

for i = 1:200
    imtemp = img(i,:);
    imtemp = reshape(imtemp,890,1280);
    result{i,3} = imtemp; 
end

%% show the result
% pl = randi(200);
% %imshowpair(result{pl,1},result{pl,2});
% imshowpair(result{pl,2},result{pl,1},'montage')
% imshowpair(result{pl,1},result{pl,2},'ColorChannels','red-cyan');
%
pl = randi(200);
subplot(2,2,[1 2])
montage([result{pl,1},result{pl,3},result{pl,2}])

subplot(2,2,3)
newImg = cat(3,result{pl,1},0.5*result{pl,3},uint8(100*ones(890,1280)));
imshow(newImg)

subplot(2,2,4)
newImg = cat(3,result{pl,2},0.5*result{pl,3},uint8(100*ones(890,1280)));
imshow(newImg)


%%
%save original image
oridir = '/home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/labels/ori';
cd(oridir);
for i = 1:200
    imt = result{i,3};
    imagename = [num2str(i,'%03d'),'.png']
    imwrite(imt,imagename);
end

%%
% save all unet result
unetlb = '/home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/labels/Unet1';
cd(unetlb);
for i = 1:200
    imt = result{i,2};
    imagename = [num2str(i,'%03d'),'.png']
    imwrite(imt,imagename);
end


%%
% save all unet result
unetlb2 = '/home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/labels/Unet2';
cd(unetlb2);
for i = 1:200
    imt = cat(3,result{i,2},0.5*result{i,3},uint8(100*ones(890,1280)));
    imagename = [num2str(i,'%03d'),'.png']
    imwrite(imt,imagename);
end

%%
function result = stitching6(path)

cd(path)
% read all mask
resultemp = cell(200,6);
result = cell(200,1);
for i = 1:200   
    for j = 1:6
        idx = (i-1)*6+(j-1);
        im = imread([num2str(idx),'_predict.png']);
        im = imresize(im,[445,445],'bicubic');
        resultemp{i,j} = im;
    end
end
% stitching all 6 pictures
for i = 1:200
    top = [1:3];
    btn = [4:6];
    imtop = maskcombine_3(resultemp(i,top));
    imbtn = maskcombine_3(resultemp(i,btn));
    result{i} = [imtop;imbtn];
    i
end
end


function result = maskcombine_3(X)
result = zeros(445,1280);
[x1,x2,x3] = deal(X{:});
result(:,1:418) = x1(:,1:418);
result(:,419:445) = (x1(:,419:445)+x2(:,1:27))./2;
result(:,446:836) = x2(:,28:418);
result(:,837:863) = (x2(:,419:445)+x3(:,1:27))./2;
result(:,864:1280) = x3(:,29:445);
result = uint8(result+50);
end

function X = cut6patch(img)
basicsize = 890/2;
cutx = [1,418,836];
cuty = [1,446];
cutp1 = [cutx(1) cuty(1) basicsize basicsize];
cutp2 = [cutx(2) cuty(1) basicsize basicsize];
cutp3 = [cutx(3) cuty(1) basicsize basicsize];
cutp4 = [cutx(1) cuty(2) basicsize basicsize];
cutp5 = [cutx(2) cuty(2) basicsize basicsize];
cutp6 = [cutx(3) cuty(2) basicsize basicsize];

X = {imcrop(img,cutp1),imcrop(img,cutp2),imcrop(img,cutp3)...
    ,imcrop(img,cutp4),imcrop(img,cutp5),imcrop(img,cutp6)};
end