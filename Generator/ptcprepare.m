% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/06/13
% Southwest university

close all,clc
clear;
cd F:\mproject\datasets\ori
% save all images in one matfile named origimg.mat
% dirOutput=dir('*.tif');
% fileNames={dirOutput.name}';
% filenum = numel(fileNames);
% for i = 1:filenum
%     im = imread(fileNames{i});
%     [m,n]=size(im);
%     if m == 480
%         im = imresize(im,[960,1280],'nearest');
%         im2 = imresize(im,[960,1280],'nearest');
%         im3 = imresize(im,[960,1280],'bilinear');
%         im4 = imresize(im,[960,1280],'bicubic');
%         im = 0.5*im2 + im3*0.3 + im4*0.2;
%     end
%     im1 = im(1:890,:);
%     switch length(num2str(i))
%         case 1
%             eval(['origimg00', num2str(i), '=', 'im1', ';']);
%         case 2
%             eval(['origimg0', num2str(i), '=', 'im1', ';']);
%         case 3
%             eval(['origimg', num2str(i), '=', 'im1', ';']);
%     end
% end

%
objnum = 158;

fvar = matfile('Datasets/origimg.mat');
pvar = matfile('Datasets/ptcell.mat');
pvar.Properties.Writable = true;
varlist = who(fvar);
disp(length(objnum+1))
img = fvar.(varlist{objnum+1});
img = im2double(img);
ptcnum = 5;
imshow(img);

%% % seedprapare
% ptcs = struct('ptc1',[],'m1',[],...
%     'ptc2',[],'m2',[],...
%     'ptc3',[],'m3',[],...
%     'ptc4',[],'m4',[],...
%     'ptc5',[],'m5',[]);
% segmente 5 particle
 close(figure(gcf));
for i = 1:ptcnum
    figure,imshow(img);
    set(gcf,'outerposition',get(0,'screensize'));
    target_rect = imrect;
    target_pos = round(getPosition(target_rect));
    % crop from original image
    target = imcrop(img,target_pos);
    delete(target_rect);close(figure(gcf));
    figure,imshow(target);
    set(gcf,'outerposition',get(0,'screensize'));
    % draw a poly to cover the target particle region
    roi_poly = impoly;
    roi_mask = createMask(roi_poly);
    seg = double(roi_mask).*target;
%     seg_props = regionprops(double(roi_mask),{'Centroid'});
%     pos_center = seg_props.Centroid;
%     pos_val = getPosition(roi_poly);
%     pos_r = round(pos_val);
%     pos = zeros(size(seg));
%     for j = 1:length(pos_r)
%         pos(pos_r(j,2),pos_r(j,1))=256-j;
%     end
%     [posm,posn] = find(pos==max(pos(:)));
%     pos([posm-1:posm+1],[posn-1:posn+1])=255;
%     
%     % kernel = [0 0.2 0;0.2 0.8 0.2;0 0.2 0];
%     % pos = im2double(conv2(pos,kernel,'same'));
    delete(roi_poly);
    close(figure(gcf));
    clearvars target_rect roi_poly
    ptc = seg;
    ptc_mask = double(roi_mask);
    
    pn = ['ptc',num2str(i)];
    mn = ['m',num2str(i)];
    
    pvar.ptcell(objnum,2*i-1) = {ptc};
    pvar.ptcell(objnum,2*i) = {ptc_mask};
    
    figure,imshowpair(ptc,double(ptc_mask),'montage');
    close(figure(gcf));
end



%%

%adding illumination

% Ig1 = imfilter(mask,fspecial('gaussian',[3 3],3),'same');
% Ig2 = imfilter(mask,fspecial('gaussian',[5 5],5),'same');
% Ig3 = imfilter(mask,fspecial('gaussian',[7 7],7),'same');
% Ig4 = imfilter(mask,fspecial('gaussian',[9 9],9),'same');
% Ig5 = imfilter(mask,fspecial('gaussian',[11 11],11),'same');
% Ig = 0.2*Ig1+0.2*Ig2+0.2*Ig3+0.2*Ig4+0.2*Ig5;
% 
% x1 = im2double(x1);
% x2 = mat2gray(x1-Ig);
% x3 = imadjust(x2,[0 0.9],[0,0.8],0.8);
% x4 =0.5*x3+0.5*x1;
% 
% 
% x1 = uint8(round(255*x1));
% x4 = uint8(round(255*x4));
% figure;
% subplot(1,3,1),imshow(x1)
% subplot(1,3,2),imshow(x4)
% subplot(1,3,3),imshow(x0)
%%
clear,clc;
load('Datasets/origimg.mat');
load('Datasets/ptcell.mat');

%
aaa = cellfun(@isempty,ptcell(:,1));
emptycell = find(aaa==1)
%
ptcell1 = ptcell;
ptcell1(emptycell,:)=[];
%
fileNames(emptycell,:)=[];

%

for i = 1:206
    if any(i==emptycell)
        disp('deleting')
    else
        switch length(num2str(i))
            case 1
                im = eval(['origimg00', num2str(i)]);
            case 2
                im = eval(['origimg0', num2str(i)]);
            case 3
                im = eval(['origimg', num2str(i)]);
        end
        im1 = reshape(im,1,[]);
        
        if i==1
            img = im1;
        else
            img = [img;im1];
        end
    end
end

%%
svar = matfile('Datasets/synimg.mat');
svar.Properties.Writable = true;

