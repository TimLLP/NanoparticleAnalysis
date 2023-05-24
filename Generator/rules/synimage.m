function [map,map_mask]= synimage(circles,imbg,imbg_mask,ptc,ptc_mask)
% make a synthesized image
%%%%%% input:
% circles: the parameter
% imbg,imbg_mask: the initial canvas
% ptc,ptc_mask:the selected seed particle and its mask
%%%%%% output:
% map:the synthesized image (uint8)
% map_mask:its mask or label (uint16)

[m,n] = size(imbg);
% map and its mask initialization in GPU
map_mask = gpuArray(imbg_mask);
map = gpuArray(imbg);

% shift due to expanding the canvas 
circles(:,1) = circles(:,1)+200;
circles(:,2) = circles(:,2)+200;

for i = 1:size(circles,1)
    %%%%%%%%%%%%%%%%%%%%%% proess on the seed %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % light change
    ptcl = ptc;
    ptcarea = find(ptc~=0);
    ptcl(ptcarea) = ptc(ptcarea)-0.01*circles(i,7);
    ptc = ptcl;
    % image scale
    ptc_temp = imresize(ptc,circles(i,4),'nearest');
    ptc_mask_temp = imresize(ptc_mask,circles(i,4),'nearest');
    % image rotate
    ptc_temp = imrotate(ptc_temp,circles(i,5),'crop');
    ptc_mask_temp = imrotate(ptc_mask_temp,circles(i,5),'crop');
    % whether fliped or not
    if circles(i,6)
        ptc_temp = flip(ptc_temp,circles(i,6));
        ptc_mask_temp = flip(ptc_mask_temp,circles(i,6));
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%% placing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % loading everything in GPU
    imbg = gpuArray(imbg);
    imbg_mask = gpuArray(imbg_mask);
    ptc_temp = gpuArray(ptc_temp);
    ptc_mask_temp = gpuArray(ptc_mask_temp);
    
    % generate a cluster image
    [m_temp,n_temp] = size(ptc_temp);
    c_temp = round(circles(i,[1,2]));
    ltopx = c_temp(1)-round(n_temp/2);
    ltopy = c_temp(2)-round(m_temp/2);
    % New blank layer
    map_temp = gpuArray(zeros(m,n));
    map_mask_temp = gpuArray(uint16(zeros(m,n)));
    
    % map layer
    map_temp([ltopx:ltopx+m_temp-1],[ltopy:ltopy+n_temp-1]) = ptc_temp;
    map(find(map_temp~=0)) = map_temp(find(map_temp~=0));
    
    % map mask layer
    map_mask_temp([ltopx:ltopx+m_temp-1],[ltopy:ltopy+n_temp-1]) = ptc_mask_temp;
    map_mask(find(map_temp~=0)) = uint16(i);
    
    % shading layer
    if circles(i,8)==1
        shadL_temp = shadinglayer(map_mask_temp);
        shadL_temp(find(map_mask_temp~=0)) = 0;
        map = map-0.25*double(shadL_temp);
    end
    
end
map = gather(imcrop(map,[201,201,n-401,m-401]));
map_mask = gather(imcrop(map_mask,[201,201,n-401,m-401]));
end

function sd_mask = shadinglayer(mask)
global ptcradius
kernel_s = ceil(ptcradius);
mask = gpuArray(mask);
sh_b1 = double(edge(mask));
sh_b2 = imfilter(sh_b1,fspecial('gaussian',[9 9],9),'same');
sh_b = sh_b2;
sh_b(find(sh_b2~=0))=1;
Ig1 = imfilter(sh_b,fspecial('gaussian',[9 9],9),'same');
Ig2 = imfilter(sh_b,fspecial('gaussian',[kernel_s-10 kernel_s-10],kernel_s-10),'same');
Ig3 = imfilter(sh_b,fspecial('gaussian',[kernel_s kernel_s],kernel_s),'same');
Ig4 = imfilter(sh_b,fspecial('gaussian',[3*kernel_s+10 3*kernel_s+10],3*kernel_s+10),'same');
Ig5 = imfilter(sh_b,fspecial('gaussian',[101 101],101),'same');
co = [max(Ig5(:)),max(Ig4(:)),max(Ig3(:)),max(Ig2(:)),max(Ig1(:))];
co = co/sum(co);
Ig = co(1)*Ig1+co(2)*Ig2+co(3)*Ig3+co(4)*Ig4+co(5)*Ig5;
%sd_mask = gather(Ig);
sd_mask = Ig;
end