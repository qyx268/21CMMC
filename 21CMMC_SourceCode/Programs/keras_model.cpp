#include "keras_model.hpp"
#include "keras_model.h"

extern "C" {
	double FcollzX_val_emulator(double f_star10_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm){
		sample_FcollzX_val->set_data1d({f_star10_norm,alpha_star_norm,sigma_8_norm,redshift_norm});
    	return pow(10., FcollzX_emu.compute_output(sample_FcollzX_val)[0]);
	}
}
