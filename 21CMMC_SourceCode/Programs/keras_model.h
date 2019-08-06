#ifdef __cplusplus
extern "C" {
#endif

double FcollzX_val_emulator(double f_star10_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm);
double Fcollz_val_emulator(double f_star10_norm, double alpha_star_norm, double f_esc10_norm, double alpha_esc_norm, double sigma_8_norm, double redshift_norm);

double FcollzX_val_MINI_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mturn);
double Fcollz_val_MINI_emulator(double f_star7_mini_norm, double alpha_star_norm, double sigma_8_norm, double redshift_norm, double log10_Mturn);

#ifdef __cplusplus
}
#endif
