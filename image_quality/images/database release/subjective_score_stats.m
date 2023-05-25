load scores1
scores_jpeg1=scores;
mmt_jpeg1=mmt;
br_jpeg1=br;
load scores2
scores_jpeg2=scores;
mmt_jpeg2=mmt;
br_jpeg2=br;

% calculate mean(rmse(image_scores))
for i=1:size(scores_jpeg1,1)
    rmse_jpeg1(i)=sqrt(mean((nonzeros(scores_jpeg1(i,:))-mmt_jpeg1(i)).^2));
%    rmse_jpeg1(i)=std(nonzeros(scores_jpeg1(i,:))); 
end
for i=1:size(scores_jpeg2,1)
    rmse_jpeg2(i)=sqrt(mean((nonzeros(scores_jpeg2(i,:))-mmt_jpeg2(i)).^2));
%    rmse_jpeg2(i)=std(nonzeros(scores_jpeg2(i,:))); 
end

rmsejpeg=[rmse_jpeg1 rmse_jpeg2];
br=[br_jpeg1 br_jpeg2];
orgs=(br==0);
orgs(1:116)=0; % exclude original images from the second study..
%orgs(117:end)=0;
mean(rmsejpeg(orgs==0)) %mean RMSE over images

