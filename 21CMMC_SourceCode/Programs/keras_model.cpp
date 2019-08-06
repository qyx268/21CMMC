#include "keras_model.hpp"
#include "keras_model.h"

#define NSFR_high_KERAS (int) 50
#define NSFR_low_KERAS (int) 50
#define LOG10MTURN_NUM_KERAS (int) 50
#define LOG10MTURN_MIN_KERAS (double) 5.-9e-8
#define LOG10MTURN_MAX_KERAS (double) 10.

double LOG10MTURN_INT_KERAS = (double) ((LOG10MTURN_MAX_KERAS+9e-8 - LOG10MTURN_MIN_KERAS)) / ((double) (LOG10MTURN_NUM_KERAS - 1.));

extern "C" {
	double FcollzX_val_emulator(double f_star10_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm){
		sample_FcollzX_val->set_data1d({f_star10_norm,alpha_star_norm,sigma_8_norm,redshift_norm});
    	return pow(10., FcollzX_emu.compute_output(sample_FcollzX_val)[0]);
	}

	double Fcollz_val_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm){
		sample_Fcollz_val->set_data1d({f_star10_norm,alpha_star_norm,f_esc10_norm,alpha_esc_norm,sigma_8_norm,redshift_norm});
    	return pow(10., Fcollz_emu.compute_output(sample_Fcollz_val)[0]);
	}

	double FcollzX_val_MINI_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mturn){
		sample_FcollzX_val_MINI->set_data1d({f_star7_mini_norm,alpha_star_norm,sigma_8_norm,redshift_norm});
		std::vector<double> result = FcollzX_MINI_emu.compute_output(sample_FcollzX_val_MINI);
		int log10_Mturn_int = (int)floor( ( log10_Mturn - LOG10MTURN_MIN_KERAS) / LOG10MTURN_INT_KERAS);
		double log10_Mturn_table = LOG10MTURN_MIN_KERAS + LOG10MTURN_INT_KERAS * (double)log10_Mturn_int;
		return pow(10., result[log10_Mturn_int] + (log10_Mturn - log10_Mturn_table) / LOG10MTURN_INT_KERAS * (result[log10_Mturn_int+1] - result[log10_Mturn_int]));
	}

	// At this moment, Fcollz_MINI is the same as FcollzX_MINI
	double Fcollz_val_MINI_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mturn){
		return FcollzX_val_MINI_emulator(f_star7_mini_norm, alpha_star_norm, sigma_8_norm, redshift_norm, log10_Mturn);
		//sample_Fcollz_val_MINI->set_data1d({f_star7_mini_norm,alpha_star_norm,sigma_8_norm,redshift_norm});
		//double result[LOG10MTURN_NUM_KERAS] = FcollzX_MINI_emu.compute_output(sample_Fcollz_val_MINI);
		//int log10_Mturn_int = (int)floor( ( log10_Mturn - LOG10MTURN_MIN_KERAS) / LOG10MTURN_INT_KERAS);
		//double log10_Mturn_table = LOG10MTURN_MIN_KERAS + LOG10MTURN_INT_KERAS * (double)log10_Mturn_int;
		//return pow(10., result[log10_Mturn_int] + (log10_Mturn - log10_Mturn_table) / LOG10MTURN_INT_KERAS * (result[log10_Mturn_int+1] - result[log10_Mturn_int]));
	}

}
