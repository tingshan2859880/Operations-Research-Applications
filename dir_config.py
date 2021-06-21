import os


class DirConfig:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'old_data', 'output')
    input_path = os.path.join(base_dir, 'old_data', "input")
    raw_path = os.path.join(base_dir, 'old_data', 'raw_data')
    traffic_path = os.path.join(input_path, "traffic")
    new_path = os.path.join(base_dir, 'data', "input")
    new_output_path = os.path.join(base_dir, 'data', "output")
    src_path = os.path.join(base_dir, 'src')
    fig_path = os.path.join(new_output_path, "fig")
    dp_fig_path = os.path.join(fig_path, "dp")

    def to_new_file(self, file_name):
        return os.path.join(self.new_path, file_name)

    def to_new_output_file(self, file_name):
        return os.path.join(self.new_output_path, file_name)

    def to_input_file(self, file_name):
        return os.path.join(self.input_path, file_name)

    def to_output_file(self, file_name):
        return os.path.join(self.output_path, file_name)

    def to_raw_data_file(self, file_name):
        return os.path.join(self.raw_path, file_name)

    def to_traffic_folder(self, channel):
        return os.path.join(self.traffic_path, channel)

    def to_traffic_file(self, channel, file_name):
        """
        蝦皮 / MOMO / YAHOO
        """
        return os.path.join(self.traffic_path, channel, file_name)
    def to_dp_fig_file(self, file_name):
        return os.path.join(self.dp_fig_path, file_name)
