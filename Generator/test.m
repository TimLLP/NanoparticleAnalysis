%
clc, clear, close all;
load('zs2.mat')

zs = [zs_0; zs_1];
zs_min = min(zs);
zs_max = max(zs);
zs_ = (zs-repmat(zs_min,8400,1))./repmat((zs_max-zs_min),8400,1);
zs_0 = zs_(1:400,:);
zs_1 = zs_(1:400,:);


disl2_0 = zeros(400,400);
discos_0 = zeros(400,400);
for i = 1:400
    for j = 1:400
        temp1 = zs_0(i,:);
        temp2 = zs_0(j,:);
        disl2_0(i,j) = pdist2(temp1,temp2,'euclidean');
        discos_0(i,j) = pdist2(temp1,temp2,'cosin');
    end
end

% 
% figure
% subplot(1,2,1),imshow(disl2_0,[]);colormap('jet');
% subplot(1,2,2),imshow(discos_0,[]);colormap('jet');

%%

countl2 = topndis(disl2_0);
countcos = topndis(discos_0);

countl2_ = countl2;
countcos_ = countcos;

li = {};
cat = 1;
for i = 1:400
    for j = i:400
        if countl2_(i,j)>=5
            li{cat} = [li{cat},countl2_];
            
        end
    end
    cat = cat+1;
end
        


%%

function count = topndis(dis)
topk = 10;
[m,n] = size(dis);
if m==n
    result = zeros(m,topk);
end
% select topk sample
for i = 1:m
    A = dis(i,:);
    [B,I] = mink(A,topk);
    result(i,:) = I;
end
%finding the related groups
count = zeros(m,n);
for i = 1:m
    temp1 = result(i,:);
    for j = 1:m
        temp2 = result(j,:);
        cnt = sum(ismember(temp2,temp1));
        count(i,j) = cnt;
    end
end
end









