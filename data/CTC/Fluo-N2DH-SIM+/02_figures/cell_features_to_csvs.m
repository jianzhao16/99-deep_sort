% === Load the .mat file ===
load('Cell_Features.mat');  % This loads Cell_Features (1 x N cell array)

% === Create output folder ===
output_dir = fullfile(pwd, 'seg_csv_features');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% === Loop through each frame ===
num_frames = length(Cell_Features);

for frame_idx = 1:num_frames
    cell_array = Cell_Features{frame_idx};  % Nx1 struct array

    if isempty(cell_array)
        continue;
    end

    num_cells = length(cell_array);
    csv_data = [];

    for i = 1:num_cells
        try
            c = cell_array(i);

            centroid = c.Centroid;
            area = c.Area;
            perimeter = c.Perimeter;
            circularity = c.Circularity;
            bbox = c.BoundingBox;
            img_shape = size(c.Image);

            % Construct feature row
            row = [i, centroid(1), centroid(2), area, perimeter, circularity, ...
                   bbox(1), bbox(2), bbox(3), bbox(4), ...
                   img_shape(1), img_shape(2)];

            csv_data = [csv_data; row];
        catch
            warning("⚠️ Skipped cell %d in frame %d due to error.", i, frame_idx);
        end
    end

    % Define header
    header = {'id', 'centroid_x', 'centroid_y', 'area', 'perimeter', 'circularity', ...
              'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'img_height', 'img_width'};

    % File name
    csv_filename = sprintf('frame_%03d.csv', frame_idx - 1);  % 0-based indexing
    csv_path = fullfile(output_dir, csv_filename);

    % Write CSV with header
    fid = fopen(csv_path, 'w');
    fprintf(fid, '%s,', header{1:end-1});
    fprintf(fid, '%s\n', header{end});
    fclose(fid);

    dlmwrite(csv_path, csv_data, '-append');
    fprintf('✅ Saved %s with %d cells\n', csv_filename, size(csv_data, 1));
end
