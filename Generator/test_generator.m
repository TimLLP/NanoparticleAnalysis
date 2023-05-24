% synthetic nano-particle cluster image generation
% Author: Wang Yunfeng
% Date: 2020/08/13
% Southwest university

dbstop if error
cd /home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/Datasets
addpath /home/hadmatter/Desktop/homogeneous_nanoparitcle_cluster/generator

close all,clc,clear;

imnum = 8; % 1-200

% load all files

global plotshow

plotshow = 0;
dvar = matfile('data.mat');
cd /media/hadmatter/Adamberg/Code/homogeneous_nanoparitcle_cluster/Datasets/Datasets
% enter the corresponding folder
cd(['img',num2str(imnum,'%03d')]);
% load original img
img = reshape(dvar.img(imnum,:),890,1280);
[ptcnum,meanRadius]= ptcplot(img,dvar,imnum); %show all the particle imgs
prompt = '选择第几个ptc:ptc';
ptcnum = input(prompt);

ptc = cell2mat(dvar.ptcell(imnum,2*ptcnum-1));
ptc_mask = cell2mat(dvar.ptcell(imnum,2*ptcnum));
% return the selected 
props = regionprops(ptc_mask,'centroid');
centre = props.Centroid;
b = bwboundaries(ptc_mask);b = b{1};
global ptcradius
ptcradius = mean(sqrt((b(:,1)-centre(2)).^2 + (b(:,2)-centre(1)).^2));

%%
%%%%%%%%%% variables
numBubbles = 183;  
minRadius=50;      
maxRadius=60;      
gap = 5; 
light = 10;
shading = 1;
option = 1; 

var1 = sort(round(numBubbles+1*randn(20,1)),'descend');
var2 = sort(round(minRadius+1*randn(20,1)));
var3 = sort(round(maxRadius+1*randn(20,1)));
var4 = sort(round(gap+1*randn(20,1)));
var5 = light*ones(20,1);
var6 = shading*ones(20,1);
var7 = repmat([1;2;3;1;2],4,1);
Vars = [var1,var2,var3,var4,var5,var6,var7];
clearvars var1 var2 var3 var4 var5 var6 var7


%%ng
for ii = 1
%% all input variables

numBubbles = Vars(ii,1);  % estimated particle number
minRadius=Vars(ii,2);      % minimum radius
maxRadius=Vars(ii,3);      % maximum radius
gap = Vars(ii,4);
light = Vars(ii,5);
shading = Vars(ii,6);
option = Vars(ii,7);
%% all the depending parameter before synthesizing

%%%%%%%%%%%% 3 different types of placement
switch option
    case 1
        bubblecircle1 = bubbleSimulator(numBubbles,minRadius,maxRadius,gap);
    case 2 
        bubblecircle2 = CircleSimulator(numBubbles,minRadius,maxRadius,gap);
    case 3
        bubblecircle3 = randomSimulator(numBubbles,minRadius,maxRadius);
end
eval(['bubblecircle = bubblecircle',num2str(option),';'])
cnum = length(bubblecircle.radius);
circles = zeros(cnum,3);

if ptcradius < 20
    shading = 0;
end

%%%%%%%%% parameter table
circles(:,1) = bubblecircle.xpos;   % col 1: x-coordinate of the center position
circles(:,2) = bubblecircle.ypos; % col 2: y-coordinate of the center position
circles(:,3) = bubblecircle.radius;   % col 3: radius
circles(:,4) = circles(:,3)/ptcradius; % col 4: the scale factor
circles(:,5) = randi([-90 90],cnum,1); % col 5: rotatation fator
circles(:,6) = randi([0 2],cnum,1); % col 6: flip horizontal or flip vertical
circles(:,6) = zeros(cnum,1);
circles(:,7) = sort(0.03*normrnd(0,light,cnum,1),'descend'); % col 7: light variance
circles(:,8) = ones(cnum,1);           % col 8: whether adding shading layer
%
circlesr1 = sortrows(circles,1);
circlesr2 = sortrows(circlesr1,2);

%% generate the synthetic image
%%%%%%%%%%%% define a gaussian blur initial background as canvas
m = 890; n = 1280;
imbg = zeros(m+400,n+400);% backgrounds padding with 400 pixels
% 200 pixels for up, down, left, and right boundaries expanding
se = strel('disk',ceil(ptcradius*2));
kernel_s = round(ptcradius/2);
imgbr = imfilter(img,fspecial('gaussian', [2*kernel_s+1 2*kernel_s+1], 2*kernel_s+1),'same');
imgo = 0.5*imopen(im2double(img),se)+0.5*im2double(imgbr);
% canvas
imbg(201:end-200,201:end-200) = imgo;
imbg0 = zeros(m+400,n+400);% set a different type of backgrounds
imbg_mask = uint16(zeros(m+400,n+400));

tic
[map,map_mask] = synimage(circlesr2,imbg,imbg_mask,ptc,ptc_mask);
toc
mpkernel = 3;
map2 = imfilter(map,fspecial('gaussian',[mpkernel mpkernel],mpkernel),'same');

%%
figure(ii);
subplot(2,3,[1 4]),imshow(img)
subplot(2,3,2),imshow(map)
subplot(2,3,3),imshow(map2)
subplot(2,3,[5,6]),imshowpair(map,map_mask,'montage')

disp(['第',num2str(ii),'张生成完毕']);


end


%%
function [ptcnum,meanRadius] = ptcplot(img,dvar,num)
% show all the particle imgs(2~5) and its mask in one image
pvarind = cellfun(@isempty,dvar.ptcell(num,:));
plotnum = length(find(pvarind==0));
ptcnum = randi([1,plotnum/2],1);
meanRadius = 0;
figure;
set(gcf,'outerposition',get(0,'screensize'));
for i = 1:plotnum
    if mod(i,2)==0
        j = (i+plotnum)/2;
        subplot(5,plotnum/2,j)
        ptc_mask = cell2mat(dvar.ptcell(num,i));% ptc_mask
        props = regionprops(ptc_mask,'centroid');
        centre = props.Centroid;
        b = bwboundaries(ptc_mask);b = b{1};
        dists = sqrt((b(:,1)-centre(2)).^2 + (b(:,2)-centre(1)).^2);
        %calculate every ptc's Radius
        meanRadius = meanRadius + mean(dists);
        imshow(ptc_mask)
        title(['ptcmask ',num2str(j-plotnum/2)]);
    else
        j = (i+1)/2;
        subplot(5,plotnum/2,j)
        imshow(cell2mat(dvar.ptcell(num,i)))%ptc
        title(['ptc ',num2str(j)]);
    end
    subplot(5,plotnum/2,[plotnum+1:plotnum*5/2])
    imshow(img)
    title(['original image']);
end
meanRadius = meanRadius*2/plotnum;% average radius of the particle
end
