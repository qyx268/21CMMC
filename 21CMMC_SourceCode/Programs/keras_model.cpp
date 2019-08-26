#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
using namespace std;

namespace keras
{
	std::vector<double> read_1d_array(std::ifstream &fin, int cols);

	class DataChunk;
	class DataChunkFlat;

	class Layer;
	class LayerActivation;
	class LayerDense;

	class KerasModel;
}

class keras::DataChunk {
public:
  virtual ~DataChunk() {}
  virtual size_t get_data_dim(void) const { return 0; }
  virtual std::vector<double> const & get_1d() const { throw "not implemented"; };
  virtual std::vector<std::vector<std::vector<double> > > const & get_3d() const { throw "not implemented"; };
  virtual void set_data(std::vector<std::vector<std::vector<double> > > const &) {};
  virtual void set_data(std::vector<double> const &) {};
  //virtual unsigned int get_count();
  virtual void read_from_file(const std::string &fname) {};
  virtual void set_data1d(const std::vector<double> input) {};
  virtual void show_name() = 0;
  virtual void show_values() = 0;
};


class keras::DataChunkFlat : public keras::DataChunk {
public:
  DataChunkFlat(size_t size) : f(size) { }
  DataChunkFlat(size_t size, double init) : f(size, init) { }
  DataChunkFlat(void) { }

  std::vector<double> f;
  std::vector<double> & get_1d_rw() { return f; }
  std::vector<double> const & get_1d() const { return f; }
  void set_data(std::vector<double> const & d) { f = d; };
  size_t get_data_dim(void) const { return 1; }

  void show_name() {
    std::cout << "DataChunkFlat " << f.size() << std::endl;
  }

  void show_values() {
    std::cout << "DataChunkFlat values:" << std::endl;
    for(size_t i = 0; i < f.size(); ++i) std::cout << f[i] << " ";
    std::cout << std::endl;
  }

  void set_data1d(const std::vector<double> input);
  //unsigned int get_count() { return f.size(); }
};

class keras::Layer {
public:
  virtual void load_weights(std::ifstream &fin) = 0;
  virtual keras::DataChunk* compute_output(keras::DataChunk*) = 0;

  Layer(std::string name) : m_name(name) {}
  virtual ~Layer() {}

  virtual unsigned int get_input_rows() const = 0;
  virtual unsigned int get_input_cols() const = 0;
  virtual unsigned int get_output_units() const = 0;

  std::string get_name() { return m_name; }
  std::string m_name;
};


class keras::LayerActivation : public Layer {
public:
  LayerActivation() : Layer("Activation") {}
  void load_weights(std::ifstream &fin);
  keras::DataChunk* compute_output(keras::DataChunk*);

  virtual unsigned int get_input_rows() const { return 0; } // look for the value in the preceding layer
  virtual unsigned int get_input_cols() const { return 0; } // same as for rows
  virtual unsigned int get_output_units() const { return 0; }

  std::string m_activation_type;
};

class keras::LayerDense : public Layer {
public:
  LayerDense() : Layer("Dense") {}

  void load_weights(std::ifstream &fin);
  keras::DataChunk* compute_output(keras::DataChunk*);
  std::vector<std::vector<double> > m_weights; //input, neuron
  std::vector<double> m_bias; // neuron

  virtual unsigned int get_input_rows() const { return 1; } // flat, just one row
  virtual unsigned int get_input_cols() const { return m_input_cnt; }
  virtual unsigned int get_output_units() const { return m_neurons; }

  int m_input_cnt;
  int m_neurons;
};

class keras::KerasModel {
public:
  KerasModel(const std::string &input_fname, bool verbose);
  ~KerasModel();
  std::vector<double> compute_output(keras::DataChunk *dc);

  unsigned int get_input_rows() const { return m_layers.front()->get_input_rows(); }
  unsigned int get_input_cols() const { return m_layers.front()->get_input_cols(); }
  int get_output_length() const;

private:

  void load_weights(const std::string &input_fname);
  int m_layers_cnt; // number of layers
  std::vector<Layer *> m_layers; // container with layers
  bool m_verbose;

};


std::vector<double> keras::read_1d_array(std::ifstream &fin, int cols) {
  vector<double> arr;
  double tmp_double;
  char tmp_char;
  fin >> tmp_char; // for '['
  for(int n = 0; n < cols; ++n) {
    fin >> tmp_double;
    arr.push_back(tmp_double);
  }
  fin >> tmp_char; // for ']'
  return arr;
}

void keras::DataChunkFlat::set_data1d(const std::vector<double> input){
	f = input;
}

void keras::LayerActivation::load_weights(std::ifstream &fin) {
  fin >> m_activation_type;
  //cout << "Activation type " << m_activation_type << endl;
}

void keras::LayerDense::load_weights(std::ifstream &fin) {
  fin >> m_input_cnt >> m_neurons;
  double tmp_double;
  char tmp_char = ' ';
  for(int i = 0; i < m_input_cnt; ++i) {
    vector<double> tmp_n;
    fin >> tmp_char; // for '['
    for(int n = 0; n < m_neurons; ++n) {
      fin >> tmp_double;
      tmp_n.push_back(tmp_double);
    }
    fin >> tmp_char; // for ']'
    m_weights.push_back(tmp_n);
  }
  //cout << "weights " << m_weights.size() << endl;
  fin >> tmp_char; // for '['
  for(int n = 0; n < m_neurons; ++n) {
    fin >> tmp_double;
    m_bias.push_back(tmp_double);
  }
  fin >> tmp_char; // for ']'
  //cout << "bias " << m_bias.size() << endl;

}

keras::KerasModel::KerasModel(const string &input_fname, bool verbose)
                                                       : m_verbose(verbose) {
  load_weights(input_fname);
}

keras::DataChunk* keras::LayerActivation::compute_output(keras::DataChunk* dc) {

  vector<double> y = dc->get_1d();
  if(m_activation_type == "relu") {
    for(unsigned int k = 0; k < y.size(); ++k) {
      if(y[k] < 0) y[k] = 0;
    }
  } else if(m_activation_type == "elu") {
    for(unsigned int k = 0; k < y.size(); ++k) {
      if(y[k] < 0) y[k] = exp(y[k]) - 1.;
    }
  }

  keras::DataChunk *out = new DataChunkFlat();
  out->set_data(y);
  return out;
}

keras::DataChunk* keras::LayerDense::compute_output(keras::DataChunk* dc) {
  //cout << "weights: input size " << m_weights.size() << endl;
  //cout << "weights: neurons size " << m_weights[0].size() << endl;
  //cout << "bias " << m_bias.size() << endl;
  size_t size = m_weights[0].size();
  size_t size8 = size >> 3;
  keras::DataChunkFlat *out = new DataChunkFlat(size, 0);
  double * y_ret = out->get_1d_rw().data();

  auto const & im = dc->get_1d();

  for (size_t j = 0; j < m_weights.size(); ++j) { // iter over input
    const double * w = m_weights[j].data();
    double p = im[j];
    size_t k = 0;
    for (size_t i = 0; i < size8; ++i) { // iter over neurons
      y_ret[k]   += w[k]   * p;          // vectorize if you can
      y_ret[k+1] += w[k+1] * p;
      y_ret[k+2] += w[k+2] * p;
      y_ret[k+3] += w[k+3] * p;
      y_ret[k+4] += w[k+4] * p;
      y_ret[k+5] += w[k+5] * p;
      y_ret[k+6] += w[k+6] * p;
      y_ret[k+7] += w[k+7] * p;
      k += 8;
    }
    while (k < size) { y_ret[k] += w[k] * p; ++k; }
  }
  for (size_t i = 0; i < size; ++i) { // add biases
    y_ret[i] += m_bias[i];
  }

  return out;
}


std::vector<double> keras::KerasModel::compute_output(keras::DataChunk *dc) {
  //cout << endl << "KerasModel compute output" << endl;
  //cout << "Input data size:" << endl;
  //dc->show_name();

  keras::DataChunk *inp = dc;
  keras::DataChunk *out = 0;
  for(int l = 0; l < (int)m_layers.size(); ++l) {
    //cout << "Processing layer " << m_layers[l]->get_name() << endl;
    out = m_layers[l]->compute_output(inp);

    //cout << "Input" << endl;
    //inp->show_name();
    //cout << "Output" << endl;
    //out->show_name();
  	//out->show_values();
    if(inp != dc) delete inp;
    //delete inp;
    inp = 0L;
    inp = out;
  }

  std::vector<double> flat_out = out->get_1d();
  //out->show_values();
  delete out;

  return flat_out;
}

void keras::KerasModel::load_weights(const string &input_fname) {
  if(m_verbose) cout << "Reading model from " << input_fname << endl;
  ifstream fin(input_fname.c_str());
  string layer_type = "";
  string tmp_str = "";
  int tmp_int = 0;

  fin >> tmp_str >> m_layers_cnt;
  if(m_verbose) cout << "Layers " << m_layers_cnt << endl;

  for(int layer = 0; layer < m_layers_cnt; ++layer) { // iterate over layers
    fin >> tmp_str >> tmp_int >> layer_type;
    if(m_verbose) cout << "Layer " << tmp_int << " " << layer_type << endl;

    Layer *l = 0L;
    if(layer_type == "Activation") {
      l = new LayerActivation();
    } else if(layer_type == "Dense") {
      l = new LayerDense();
    } else if(layer_type == "Dropout") {
      continue; // we dont need dropout layer in prediciton mode
    }
    if(l == 0L) {
      cout << "Layer is empty, maybe it is not defined? Cannot define network." << endl;
      return;
    }
    l->load_weights(fin);
    m_layers.push_back(l);
  }

  fin.close();
}

keras::KerasModel::~KerasModel() {
  for(int i = 0; i < (int)m_layers.size(); ++i) {
    delete m_layers[i];
  }
}

int keras::KerasModel::get_output_length() const
{
  int i = m_layers.size() - 1;
  while ((i > 0) && (m_layers[i]->get_output_units() == 0)) --i;
  return m_layers[i]->get_output_units();
}

using namespace keras;

// allocate the sample space and load the emulators
DataChunk *sample_Fcollz_val = new DataChunkFlat();
KerasModel Fcollz_val_emu("../Emulators/Keras/log10_Fcollz_val.txt", false);

DataChunk *sample_Fcollz_val_MINI = new DataChunkFlat();
KerasModel Fcollz_val_MINI_emu("../Emulators/Keras/log10_Fcollz_val_MINI.txt", false);

DataChunk *sample_Fcollz_val_highZ = new DataChunkFlat();
KerasModel Fcollz_val_highZ_emu("../Emulators/Keras/log10_Fcollz_val_highZ.txt", false);

DataChunk *sample_Fcollz_val_MINI_highZ = new DataChunkFlat();
KerasModel Fcollz_val_MINI_highZ_emu("../Emulators/Keras/log10_Fcollz_val_MINI_highZ.txt", false);

DataChunk *sample_log10_Fcoll_spline_SFR_low = new DataChunkFlat();
KerasModel log10_Fcoll_spline_SFR_low_emu("../Emulators/Keras/log10_Fcoll_spline_SFR_low.txt", false);

DataChunk *sample_log10_Fcoll_spline_SFR_MINI_low = new DataChunkFlat();
KerasModel log10_Fcoll_spline_SFR_MINI_low_emu("../Emulators/Keras/log10_Fcoll_spline_SFR_MINI_low.txt", false);

DataChunk *sample_Fcoll_spline_SFR_high = new DataChunkFlat();
KerasModel Fcoll_spline_SFR_high_emu("../Emulators/Keras/log10_Fcoll_spline_SFR_high.txt", false);

DataChunk *sample_Fcoll_spline_SFR_MINI_high = new DataChunkFlat();
KerasModel Fcoll_spline_SFR_MINI_high_emu("../Emulators/Keras/log10_Fcoll_spline_SFR_MINI_high.txt", false);

// TODO: This is faster than the below which has been commented
extern "C" double Fcollz_val_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm);
extern "C" double Fcollz_val_MINI_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm);
extern "C" double Fcollz_val_highZ_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm);
extern "C" double Fcollz_val_MINI_highZ_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm);
extern "C" double log10_Fcoll_spline_SFR_low_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm);
extern "C" double log10_Fcoll_spline_SFR_MINI_low_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm);
extern "C" double Fcoll_spline_SFR_high_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm);
extern "C" double Fcoll_spline_SFR_MINI_high_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm);

// this is actualy log_10 and is returning linear
double Fcollz_val_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm){
	sample_Fcollz_val->set_data1d({f_star10_norm,alpha_star_norm,f_esc10_norm,alpha_esc_norm,sigma_8_norm,redshift_norm});
	return pow(10., Fcollz_val_emu.compute_output(sample_Fcollz_val)[0]);
}

// this is actualy log_10 and is returning linear
double Fcollz_val_MINI_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm){
	sample_Fcollz_val_MINI->set_data1d({f_star7_mini_norm,alpha_star_norm,sigma_8_norm,redshift_norm, log10_Mmin_norm});
	return pow(10., Fcollz_val_MINI_emu.compute_output(sample_Fcollz_val_MINI)[0]);
}

// this is actualy log_10 and is returning linear
double Fcollz_val_highZ_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm){
	sample_Fcollz_val_highZ->set_data1d({f_star10_norm,alpha_star_norm,f_esc10_norm,alpha_esc_norm,sigma_8_norm,redshift_norm});
	return pow(10., Fcollz_val_highZ_emu.compute_output(sample_Fcollz_val_highZ)[0]);
}

// this is actualy log_10 and is returning linear
double Fcollz_val_MINI_highZ_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm){
	sample_Fcollz_val_MINI_highZ->set_data1d({f_star7_mini_norm,alpha_star_norm,sigma_8_norm,redshift_norm, log10_Mmin_norm});
	return pow(10., Fcollz_val_MINI_highZ_emu.compute_output(sample_Fcollz_val_MINI_highZ)[0]);
}
// this is actualy log_e and is returning log_e
double log10_Fcoll_spline_SFR_low_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm){
	sample_log10_Fcoll_spline_SFR_low->set_data1d({f_star10_norm, alpha_star_norm, f_esc10_norm, alpha_esc_norm, sigma_8_norm, redshift_norm, log10_Mmin_norm, R_norm, dens_norm});
	return log10_Fcoll_spline_SFR_low_emu.compute_output(sample_log10_Fcoll_spline_SFR_low)[0];
}
// this is actualy log_e and is returning log_e
double log10_Fcoll_spline_SFR_MINI_low_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm){
	sample_log10_Fcoll_spline_SFR_MINI_low->set_data1d({f_star7_mini_norm, alpha_star_norm, sigma_8_norm, redshift_norm, log10_Mmin_norm, R_norm, dens_norm});
	return log10_Fcoll_spline_SFR_MINI_low_emu.compute_output(sample_log10_Fcoll_spline_SFR_MINI_low)[0];
}
// this is actualy log_10 and is returning linear
double Fcoll_spline_SFR_high_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm){
	sample_Fcoll_spline_SFR_high->set_data1d({f_star10_norm, alpha_star_norm, f_esc10_norm, alpha_esc_norm, sigma_8_norm, redshift_norm, log10_Mmin_norm, R_norm, dens_norm});
	return pow(10., Fcoll_spline_SFR_high_emu.compute_output(sample_Fcoll_spline_SFR_high)[0]);
}
// this is actualy log_10 and is returning linear
double Fcoll_spline_SFR_MINI_high_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mmin_norm, double R_norm, double dens_norm){
	sample_Fcoll_spline_SFR_MINI_high->set_data1d({f_star7_mini_norm, alpha_star_norm, sigma_8_norm, redshift_norm, log10_Mmin_norm, R_norm, dens_norm});
	return pow(10., Fcoll_spline_SFR_MINI_high_emu.compute_output(sample_Fcoll_spline_SFR_MINI_high)[0]);
}

//extern "C" double Fcollz_val_emulator(double *input); 
//extern "C" double Fcollz_val_MINI_emulator(double *input);
//extern "C" double log10_Fcoll_spline_SFR_low_emulator(double *input);
//extern "C" double log10_Fcoll_spline_SFR_MINI_low_emulator(double *input);
//extern "C" double Fcoll_spline_SFR_high_emulator(double *input);
//extern "C" double Fcoll_spline_SFR_MINI_high_emulator(double *input);
//
//// this is actualy log_10 and is returning linear
//double Fcollz_val_emulator(double *input){
//	sample_Fcollz_val->set_data1d({input[0], input[1], input[2], input[3], input[4], input[5]});
//	return pow(10., Fcollz_val_emu.compute_output(sample_Fcollz_val)[0]);
//}
//
//// this is actualy log_10 and is returning linear
//double Fcollz_val_MINI_emulator(double *input){
//	sample_Fcollz_val_MINI->set_data1d({input[0], input[1], input[2], input[3], input[4]});
//	return pow(10., Fcollz_val_MINI_emu.compute_output(sample_Fcollz_val_MINI)[0]);
//}
//
//// this is actualy log_e and is returning log_e
//double log10_Fcoll_spline_SFR_low_emulator(double *input){
//	sample_log10_Fcoll_spline_SFR_low->set_data1d({input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8]});
//	return log10_Fcoll_spline_SFR_low_emu.compute_output(sample_log10_Fcoll_spline_SFR_low)[0];
//}
//// this is actualy log_e and is returning log_e
//double log10_Fcoll_spline_SFR_MINI_low_emulator(double *input){
//	sample_log10_Fcoll_spline_SFR_MINI_low->set_data1d({input[0], input[1], input[2], input[3], input[4], input[5], input[6]});
//	return log10_Fcoll_spline_SFR_MINI_low_emu.compute_output(sample_log10_Fcoll_spline_SFR_MINI_low)[0];
//}
//// this is actualy log_10 and is returning linear
//double Fcoll_spline_SFR_high_emulator(double *input){
//	sample_Fcoll_spline_SFR_high->set_data1d({input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7], input[8]});
//	return pow(10., Fcoll_spline_SFR_high_emu.compute_output(sample_Fcoll_spline_SFR_high)[0]);
//}
//// this is actualy log_10 and is returning linear
//double Fcoll_spline_SFR_MINI_high_emulator(double *input){
//	sample_Fcoll_spline_SFR_MINI_high->set_data1d({input[0], input[1], input[2], input[3], input[4], input[5], input[6]});
//	return pow(10., Fcoll_spline_SFR_MINI_high_emu.compute_output(sample_Fcoll_spline_SFR_MINI_high)[0]);
//}
