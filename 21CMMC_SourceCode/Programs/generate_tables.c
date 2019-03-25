#include "../Parameter_files/INIT_PARAMS.H"
#include "../Parameter_files/ANAL_PARAMS.H"
#include "../Parameter_files/Variables.h"
#include "bubble_helper_progs.c"
#include "heating_helper_progs.c"
#include "gsl/gsl_sf_erf.h"
#include "filter.c"
#include <stdlib.h>
#include <time.h>

float REDSHIFT;

void init_21cmMC_Ts_arrays();
void init_21cmMC_HII_arrays();
void ComputeTsBoxes();
void ComputeIonisationBoxes(int sample_index, float REDSHIFT_SAMPLE);
void destroy_21cmMC_Ts_arrays();
void destroy_21cmMC_HII_arrays();

int main(int argc, char ** argv){
    // The standard build of 21cmFAST requires openmp for the FFTs. 21CMMC does not, however, for some computing architectures, I found it important to include this
    omp_set_num_threads(1);
	flag_generate_tables = 1;
    
    char filename[500];
    char cmd[500];
    char dummy_string[500];
    FILE *F;
    
    int i,j,k,temp_int,temp_int2, counter;
    float z_prime,prev_z_prime;
    
    unsigned long long ct;

    INDIVIDUAL_ID = atof(argv[1]);
    INDIVIDUAL_ID_2 = atof(argv[2]);
    N_USER_REDSHIFT = atof(argv[3]);
    // ****** NOTE: Need to add in a flag here, which toggles including alpha as a useable parameter ****** //
    // ****** In doing it here, it enables the majority of the remaining code to be relatively straight-forward (i.e. doesn't change existing text-file structure etc.) ****** //
    // ****** This hasn't been rigorously checked yet. Need to look into this at some point... ******* //
    //INCLUDE_ZETA_PL = atof(argv[5]);
    USE_MASS_DEPENDENT_ZETA = atoi(argv[5]); // New in v1.4
    
    // Redshift for which Ts.c is evolved down to, i.e. z'
    REDSHIFT = atof(argv[6]);

    // New in v1.4
    // Flag set to 1 if Luminosity functions are to be used together with outputs from 21cm signals.
    // Flag set to 2 if one wants to compute Luminosity functions with tau_e, i.e. without PS
    USE_LF = atof(argv[7]);

    // Determines the lenght of the walker file, given the values set by TOTAL_AVAILABLE_PARAMS in Variables.h and the number of redshifts
    WALKER_FILE_LENGTH = TOTAL_AVAILABLE_PARAMS + 1;
    
    // Create arrays to read in all the parameter data from the two separate walker files
    double *PARAM_COSMOLOGY_VALS = calloc(TOTAL_COSMOLOGY_FILEPARAMS,sizeof(double));
    double *PARAM_VALS = calloc(TOTAL_AVAILABLE_PARAMS,sizeof(double));
    
    /////////////////   Read in the cosmological parameter data     /////////////////
    sprintf(filename,"WalkerCosmology_%1.6lf_%1.6lf.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2);
	if (F = fopen(filename,"rt")){
      for(i=0;i<TOTAL_COSMOLOGY_FILEPARAMS;i++) {
          fscanf(F,"%s\t%lf\n",&dummy_string,&PARAM_COSMOLOGY_VALS[i]);
      }
      fclose(F);
	}
	else{
		return 0;
	}
    
    // Assign these values. Hard-coded, so order is important
    RANDOM_SEED = (unsigned long long)PARAM_COSMOLOGY_VALS[0];
    SIGMA8 = (float)PARAM_COSMOLOGY_VALS[1];
    hlittle = (float)PARAM_COSMOLOGY_VALS[2];
    OMm = (float)PARAM_COSMOLOGY_VALS[3];
    OMl = (float)PARAM_COSMOLOGY_VALS[4];
    OMb = (float)PARAM_COSMOLOGY_VALS[5];
    POWER_INDEX = (float)PARAM_COSMOLOGY_VALS[6]; //power law on the spectral index, ns
    
    sprintf(filename,"Walker_%1.6lf_%1.6lf.txt",INDIVIDUAL_ID,INDIVIDUAL_ID_2);
    F = fopen(filename,"rt");
    
    temp_int = 0;
    temp_int2 = 0;
    for(i=0;i<WALKER_FILE_LENGTH;i++) {
        if(i==0) {
            fscanf(F,"%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n",&dummy_string,&GenerateNewICs,&SUBCELL_RSD,&USE_FCOLL_IONISATION_TABLE,&SHORTEN_FCOLL,&USE_TS_FLUCT,&INHOMO_RECO,&STORE_DATA);
        }
        else if (i > 0 && i <= TOTAL_AVAILABLE_PARAMS) {
            fscanf(F,"%s\t%lf\n",&dummy_string,&PARAM_VALS[temp_int]);
            temp_int += 1;
        }
        else {
             continue;
        }
    }
    fclose(F);

    // Initialise the power spectrum data, and relevant functions etc., for the entire file here (i.e. it is only done once here)
    init_ps();
    counter = 0;
    z_prime = REDSHIFT*1.0001; //higher for rounding
    while (z_prime < Z_HEAT_MAX)
        z_prime = ((1.+z_prime)*ZPRIME_STEP_FACTOR - 1.);
    prev_z_prime = Z_HEAT_MAX;
    z_prime = ((1.+z_prime)/ ZPRIME_STEP_FACTOR - 1.);
    
    while (z_prime > REDSHIFT){
        counter += 1;
        prev_z_prime = z_prime;
        z_prime = ((1.+prev_z_prime) / ZPRIME_STEP_FACTOR - 1.);
    }
    N_USER_REDSHIFT = counter;
    redshifts = calloc(N_USER_REDSHIFT,sizeof(double));
    counter = 0;
    z_prime = REDSHIFT*1.0001; //higher for rounding
    while (z_prime < Z_HEAT_MAX)
        z_prime = ((1.+z_prime)*ZPRIME_STEP_FACTOR - 1.);
    prev_z_prime = Z_HEAT_MAX;
    z_prime = ((1.+z_prime)/ ZPRIME_STEP_FACTOR - 1.);
    while (z_prime > REDSHIFT){
		redshifts[counter] = z_prime;
        counter += 1;
        prev_z_prime = z_prime;
        z_prime = ((1.+prev_z_prime) / ZPRIME_STEP_FACTOR - 1.);
    }
	LOG10MTURN_INT = (double) ((LOG10MTURN_MAX+9e-8 - LOG10MTURN_MIN)) / ((double) (LOG10MTURN_NUM - 1.));
    F_STAR10 = pow(10.,PARAM_VALS[0]);
    ALPHA_STAR = PARAM_VALS[1];
    F_ESC10 = pow(10.,PARAM_VALS[2]);
    ALPHA_ESC = PARAM_VALS[3];
    initialiseSplinedSigmaM_quicker(1e5/50.,1e20);
    R_BUBBLE_MAX = 50.;
    F_STAR10_MINI = pow(10.,PARAM_VALS[20]) * pow(1e3, ALPHA_STAR);
    F_ESC_MINI = pow(10.,PARAM_VALS[21]);
    sprintf(cmd,"mkdir -p ../InterpolationTables/Walker_%1.6lf_%1.6lf",INDIVIDUAL_ID,INDIVIDUAL_ID_2); system(cmd);
    init_MHR();
    ComputeTsBoxes();
    free(PARAM_VALS);
    free(redshifts);
    free_ps();
    free_MHR();
	sprintf(cmd,"mv ./Walker*_%1.6lf_%1.6lf.txt ../InterpolationTables/Walker_%1.6lf_%1.6lf",INDIVIDUAL_ID,INDIVIDUAL_ID_2, INDIVIDUAL_ID,INDIVIDUAL_ID_2); system(cmd);
    return 0;
}

void ComputeTsBoxes() {
	FILE *f;
	char filename[500];
    int R_ct,i,ii,j,k,i_z, counter;
    float growth_factor_z, inverse_growth_factor_z, R, R_factor, zp, prev_zp, zpp, prev_zpp, prev_R;
    float min_density = -1. + 9e-8;
	float max_density = 1.5*1.001;
	float zp_table, zpp_integrand;
    float *min_densities = calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
    float *max_densities = calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
	for (i=0;i<NUM_FILTER_STEPS_FOR_Ts;i++){
		min_densities[i] = min_density;
		max_densities[i] = max_density;
	}
    init_21cmMC_Ts_arrays();
    R = L_FACTOR*BOX_LEN/(float)HII_DIM;
    R_factor = pow(R_XLy_MAX/R, 1./(float)NUM_FILTER_STEPS_FOR_Ts);
    for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        R_values[R_ct] = R;
        R *= R_factor;
    } //end for loop through the filter scales R
    zp = REDSHIFT*1.0001; //higher for rounding
    while (zp < Z_HEAT_MAX) {
        zp = ((1.+zp)*ZPRIME_STEP_FACTOR - 1);
    }
    prev_zp = Z_HEAT_MAX;
    zp = ((1.+zp)/ ZPRIME_STEP_FACTOR - 1);
    determine_zpp_min = REDSHIFT*0.999;
    for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
        if (R_ct==0){
            prev_zpp = zp; prev_R = 0;
        }
        else{
            prev_zpp = zpp_edge[R_ct-1]; prev_R = R_values[R_ct-1];
        }
        zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
        zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
    }
    determine_zpp_max = zpp*1.001;
    zpp_bin_width = (determine_zpp_max - determine_zpp_min)/((float)zpp_interp_points_SFR-1.0);
    for (i=0; i<zpp_interp_points_SFR;i++) {
        zpp_interp_table[i] = determine_zpp_min + zpp_bin_width*(float)i;
    }
    sprintf(filename,"../InterpolationTables/Walker_%1.6lf_%1.6lf/zpp_interp_table.bin",INDIVIDUAL_ID,INDIVIDUAL_ID_2); f = fopen(filename, "wb");
	fwrite(zpp_interp_table, sizeof(float), zpp_interp_points_SFR, f); fclose(f);
    initialise_FgtrM_st_SFR_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, ALPHA_STAR, ALPHA_ESC, F_STAR10, F_ESC10, F_STAR10_MINI, F_ESC_MINI);
    initialise_Xray_FgtrM_st_SFR_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max, ALPHA_STAR, F_STAR10, F_STAR10_MINI);
    zp_table = zp;
    counter = 0;
    for (i=0; i<N_USER_REDSHIFT; i++) {
          for (R_ct=0; R_ct<NUM_FILTER_STEPS_FOR_Ts; R_ct++){
              if (R_ct==0){
                  prev_zpp = zp_table; prev_R = 0;
              }
              else{
                  prev_zpp = zpp_edge[R_ct-1]; prev_R = R_values[R_ct-1];
              }
            zpp_edge[R_ct] = prev_zpp - (R_values[R_ct] - prev_R)*CMperMPC / drdz(prev_zpp); // cell size
            zpp = (zpp_edge[R_ct]+prev_zpp)*0.5; // average redshift value of shell: z'' + 0.5 * dz''
            redshift_interp_table[counter] = zpp;
            growth_interp_table[counter] = dicke(zpp);
            Mcrit_atom_interp_table[counter] = atomic_cooling_threshold(zpp);
            counter += 1;
          }
        prev_zp = zp_table;
        zp_table = ((1.+prev_zp) / ZPRIME_STEP_FACTOR - 1);
    }
    initialise_Xray_Fcollz_SFR_Conditional_table(NUM_FILTER_STEPS_FOR_Ts,min_densities,max_densities,growth_interp_table,R_values, Mcrit_atom_interp_table, ALPHA_STAR, F_STAR10, F_STAR10_MINI);
    counter = 0;
    while (zp > REDSHIFT){
        for(i_z=0;i_z<N_USER_REDSHIFT;i_z++) {
            if(fabs(redshifts[i_z] - zp)<0.001) ComputeIonisationBoxes(i_z,redshifts[i_z]);
        }
        prev_zp = zp;
        zp = ((1.+prev_zp) / ZPRIME_STEP_FACTOR - 1);
        counter += 1;
    } // end main integral loop over z'
    destroy_21cmMC_Ts_arrays();
}

void ComputeIonisationBoxes(int sample_index, float REDSHIFT_SAMPLE) {
    int i,j,k,ii, x,y,z, N_min_cell, LAST_FILTER_STEP, first_step_R;
    int n_x, n_y, n_z,counter,LOOP_INDEX;
    float R, cell_length_factor, M_MIN, stored_R, massofscaleR;
    float Mlim_Fstar, Mlim_Fesc; // New in v1.4
	float Mlim_Fstar_MINI;
    float min_density, max_density;
	min_density = -1. + 9e-8;
	max_density = 1.5*1.001;
    xi_low = calloc((NGLlow+1),sizeof(float));
    wi_low = calloc((NGLlow+1),sizeof(float));
    xi_high = calloc((NGLhigh+1),sizeof(float));
    wi_high = calloc((NGLhigh+1),sizeof(float));
    xi_SFR = calloc((NGL_SFR+1),sizeof(float));
    wi_SFR = calloc((NGL_SFR+1),sizeof(float));
    log10_overdense_spline_SFR = calloc(NSFR_low,sizeof(double));
    Overdense_spline_SFR = calloc(NSFR_high,sizeof(float));
    int determine_R_intermediate;
    determine_R_intermediate = 0;
    R=fmax(R_BUBBLE_MIN, (cell_length_factor*BOX_LEN/(float)HII_DIM));
    int N_RSTEPS, counter_R;
    counter = 0;
    while ((R - fmin(R_BUBBLE_MAX, L_FACTOR*BOX_LEN)) <= FRACT_FLOAT_ERR ) {
        R*= DELTA_R_HII_FACTOR;
        determine_R_intermediate += 1;
        if(R >= fmin(R_BUBBLE_MAX, L_FACTOR*BOX_LEN)) {
            stored_R = R/DELTA_R_HII_FACTOR;
        }
        counter += 1;
    }
    N_RSTEPS = counter;
    counter_R = N_RSTEPS;
    determine_R_intermediate = determine_R_intermediate - 2;
    R=fmin(R_BUBBLE_MAX, L_FACTOR*BOX_LEN);
    LAST_FILTER_STEP = 0;
	M_MIN = 1e5;
    initialiseSplinedSigmaM(M_MIN,1e16);
	Mlim_Fstar = Mass_limit_bisection(M_MIN, 1e16,  ALPHA_STAR, F_STAR10);
	Mlim_Fesc = Mass_limit_bisection(M_MIN, 1e16, ALPHA_ESC, F_ESC10);
	Mlim_Fstar_MINI = Mass_limit_bisection(M_MIN, 1e16,  ALPHA_STAR, F_STAR10_MINI);
    first_step_R = 1;
    counter = 0;
    while (!LAST_FILTER_STEP && (M_MIN < RtoM(R)) ){
        // Check if we are the last filter step
        if ( ((R/DELTA_R_HII_FACTOR - cell_length_factor*BOX_LEN/(float)HII_DIM) <= FRACT_FLOAT_ERR) || ((R/DELTA_R_HII_FACTOR - R_BUBBLE_MIN) <= FRACT_FLOAT_ERR) ) {
            LAST_FILTER_STEP = 1;
            R = fmax(cell_length_factor*BOX_LEN/(double)HII_DIM, R_BUBBLE_MIN);
        }
        massofscaleR = RtoM(R);
        initialiseGL_FcollSFR(NGL_SFR, M_MIN,massofscaleR);
        initialiseFcollSFR_spline(REDSHIFT_SAMPLE,min_density,max_density,M_MIN,massofscaleR,Mturn_interp_table,ALPHA_STAR,ALPHA_ESC,F_STAR10,F_ESC10,Mlim_Fstar,Mlim_Fesc,F_STAR10_MINI,Mlim_Fstar_MINI);
        if(first_step_R) {
            R = stored_R;
            first_step_R = 0;
        }
        else
            R /= DELTA_R_HII_FACTOR;
        counter_R -= 1;
    }
    free(xi_low);
    free(wi_low);
    free(xi_high);
    free(wi_high);
    free(xi_SFR);
    free(wi_SFR);
    free(log10_overdense_spline_SFR);
    free(Overdense_spline_SFR);
    free(Mass_Spline);
    free(Sigma_Spline);
    free(dSigmadm_Spline);
    free(second_derivs_sigma);
    free(second_derivs_dsigma);
}

void init_21cmMC_Ts_arrays() {
    
    int i,j;
	char filename[500];
	FILE *f;
	//for continuously run HII calc
    log10_Fcoll_spline_SFR = calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
    Fcoll_spline_SFR = calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));

    log10_Fcoll_spline_SFR_MINI = calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
    Fcoll_spline_SFR_MINI = calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));

    for (i=0;i<NSFR_low*LOG10MTURN_NUM;i++){
        log10_Fcoll_spline_SFR[i] = 0.;
        log10_Fcoll_spline_SFR_MINI[i] = 0.;
    }
    for (i=0;i<NSFR_high*LOG10MTURN_NUM;i++){
        Fcoll_spline_SFR[i] = 0.;
        Fcoll_spline_SFR_MINI[i] = 0.;
    }
        
	Mturn_interp_table = (double *)calloc(LOG10MTURN_NUM, sizeof(double));
    for (i=0; i <LOG10MTURN_NUM; i++){
	  Mturn_interp_table[i] = pow(10., LOG10MTURN_MIN + (double)i*LOG10MTURN_INT);
	}
    sprintf(filename,"../InterpolationTables/Walker_%1.6lf_%1.6lf/Mturn_interp_table.bin",INDIVIDUAL_ID,INDIVIDUAL_ID_2); f = fopen(filename, "wb");
	fwrite(Mturn_interp_table, sizeof(double), LOG10MTURN_NUM, f); fclose(f); 

    xi_SFR_Xray = calloc((NGL_SFR+1),sizeof(float));
    wi_SFR_Xray = calloc((NGL_SFR+1),sizeof(float));

    zpp_interp_table = calloc(zpp_interp_points_SFR, sizeof(float));

    redshift_interp_table = calloc(NUM_FILTER_STEPS_FOR_Ts*Nsteps_zp, sizeof(float)); // New
    growth_interp_table = calloc(NUM_FILTER_STEPS_FOR_Ts*N_USER_REDSHIFT, sizeof(float)); // New
    Mcrit_atom_interp_table = calloc(NUM_FILTER_STEPS_FOR_Ts*N_USER_REDSHIFT, sizeof(float)); // New

    overdense_Xray_low_table = calloc(NSFR_low,sizeof(double));
    log10_Fcollz_SFR_Xray_low_table = (double ***)calloc(N_USER_REDSHIFT,sizeof(double **)); //New
    log10_Fcollz_SFR_Xray_low_table_MINI = (double ***)calloc(N_USER_REDSHIFT,sizeof(double **)); //New
    for(i=0;i<N_USER_REDSHIFT;i++){  // New
        log10_Fcollz_SFR_Xray_low_table[i] = (double **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double *));
        log10_Fcollz_SFR_Xray_low_table_MINI[i] = (double **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double *));
        for(j=0;j<NUM_FILTER_STEPS_FOR_Ts;j++) {
            log10_Fcollz_SFR_Xray_low_table[i][j] = (double *)calloc(NSFR_low,sizeof(double));
            log10_Fcollz_SFR_Xray_low_table_MINI[i][j] = (double *)calloc(NSFR_low*LOG10MTURN_NUM,sizeof(double));
        }
    }

    Overdense_Xray_high_table = calloc(NSFR_high,sizeof(float));
    Fcollz_SFR_Xray_high_table = (float ***)calloc(N_USER_REDSHIFT,sizeof(float **)); //New
    Fcollz_SFR_Xray_high_table_MINI = (float ***)calloc(N_USER_REDSHIFT,sizeof(float **)); //New
    for(i=0;i<N_USER_REDSHIFT;i++){  // New
        Fcollz_SFR_Xray_high_table[i] = (float **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
        Fcollz_SFR_Xray_high_table_MINI[i] = (float **)calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float *));
        for(j=0;j<NUM_FILTER_STEPS_FOR_Ts;j++) {
            Fcollz_SFR_Xray_high_table[i][j] = (float *)calloc(NSFR_high,sizeof(float));
            Fcollz_SFR_Xray_high_table_MINI[i][j] = (float *)calloc(NSFR_high*LOG10MTURN_NUM,sizeof(float));
        }
    }
    zpp_edge = calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(double));
    R_values = calloc(NUM_FILTER_STEPS_FOR_Ts,sizeof(float));
}

void destroy_21cmMC_Ts_arrays() {
    
    int i,j;
    
    free(log10_Fcoll_spline_SFR);
    free(Fcoll_spline_SFR);
    free(log10_Fcoll_spline_SFR_MINI);
    free(Fcoll_spline_SFR_MINI);

    free(xi_SFR_Xray);
    free(wi_SFR_Xray);

	free(Mturn_interp_table);
    
    free(zpp_interp_table);
    free(redshift_interp_table);
    free(growth_interp_table);
    
    free(overdense_Xray_low_table);
    free(Overdense_Xray_high_table);
    
    for(i=0;i<N_USER_REDSHIFT;i++){
        for(j=0;j<NUM_FILTER_STEPS_FOR_Ts;j++) {
            free(log10_Fcollz_SFR_Xray_low_table[i][j]);
            free(Fcollz_SFR_Xray_high_table[i][j]);
        }
        free(log10_Fcollz_SFR_Xray_low_table[i]);
        free(Fcollz_SFR_Xray_high_table[i]);
    }
    free(log10_Fcollz_SFR_Xray_low_table);
    free(Fcollz_SFR_Xray_high_table);

    free(R_values);
    free(zpp_edge);
}

