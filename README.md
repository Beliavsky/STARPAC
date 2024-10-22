# STARPAC
[Standards Time Series and Regression Package](https://water.usgs.gov/software/OTIS/addl/starpac/), a library of Fortran subroutines for statistical data analysis developed by the Statistical Engineering Division of the National Institute of Standards and Technology by Janet R. Donaldson and Peter V. Tryon, converted to free source form by [John Burkardt](https://people.math.sc.edu/Burkardt/f_src/starpac/starpac.html), who supplied the list of routines below. Compile with `gfortran -fallow-argument-mismatch starpac.f90 starpac_test.f90`.

```
List of Routines:

ABSCOM counts the entries of | V(1:N) - W(1:N) | greater than ABSTOL.
ACCDIG returns the number of accurate digits in an approximation to X.
ACFD computes autocorrelations and partial autocorrelations.
ACFDTL prints titles for ACORRD.
ACFER does error checking for the ACF routines.
ACF is the simple interface to the autocorrelations routines.
ACFF computes autocorrelations of a time series using an FFT.
ACFFS uses an FFT with ACVF estimates for autocorrelations of a time series.
ACFLST lists the autocorrelations and other information.
ACFM computes autocorrelations of a time series with missing data.
ACFMN computes autocorrelations of a time series.
ACFMNF computes autocorrelations of a time series.
ACFMNM computes autocorrelations of a time series with missing data.
ACFMS is the user interface for autocorrelations of a time series with missing data.
ACFOUT prints autocorrelations.
ACFSD computes the standard error of autocorrelations.
ACFSDM computes the standard error of autocorrelations with missing data.
ACFS computes autocorrelations with computed ACVF estimates.
ACVF computes the autocovariance function of a series.
ACVFF computes the ACVF of a series using two FFT passes.
ACVFM computes autocovariance when missing data is involved.
ADJLMT corrects the plot limits when all observations are equal.
AIMEC is the user interface for ARIMA estimation.
AIME is the user interface for ARIMA estimation, control call.
AIMES is the user interface for ARIMA estimation, long call.
AIMF is the user interface for ARIMA estimation, short call.
AIMFS is the user interface for ARIMA estimation, control call.
AIMX1 sets the starting parameter values for AIMX.
ALBETA computes the logarithm of the Beta function.
ALGAMS evaluates the log of the absolute value of the Gamma function.
ALNGAM computes the logarithm of the absolute value of the Gamma function.
ALNREL evaluates log ( 1 + X ) with relative error control.
AMDRV estimates the jacobian matrix.
AMEAN computes the arithmetic mean of a series.
AMEANM computes the arithmetic mean of a series with missing data.
AMECNT is the control routine for nonlinear least squares regression.
AMEDRV is the control routine for nonlinear least squares regression.
AMEER checks errors for the nonlinear least squares estimation.
AMEFIN analyzes nonlinear least squares estimates after they are computed.
AMEHDR prints headings for nonlinear least squares estimation.
AMEISM prints an initial summary for nonlinear least squares routines.
AMEMN is the control routine for using the NL2 software package.
AMEOUT prints the final summary output from ARIMA estimation.
AMEPT1 prints data summary for nonlinear least squares routines.
AMEPT2 prints four standardized residual plots.
AMESTP controls the step size selection.
AMFCNT is the control routine for ARIMA forecasting.
AMFER checks errors for nonlinear least squares estimation.
AMFHDR prints headers for nonlinear least squares estimation.
AMFMN computes and prints ARIMA forecasts.
AMFOUT produces ARIMA forecasting output.
AMLST1 prints parameters for the ARIMA routine.
AMLST prints parameter summaries from ARIMA forecasting.
AOS computes autoregressive model order selection statistics.
AOSLST lists the autoregressive model order selection statistics.
AOV1ER does preliminary checks on input to the one-way family.
AOV1 is a user interface to AOV1MN, one-way analysis of variance.
AOV1HD prints headers for the one-way ANOVA family.
AOV1MN computes results for analysis of a one-way classification.
AOV1S is a user interface for AOV1MN, one-way analysis of variance.
AOV1XP prints storage for one-way family exerciser.
ARCOEF uses Durbin's method for autoregression coefficients with order lag.
ARFLT performs autoregressive filtering.
ASSESS assesses a candidate step.
AXPBY: SZ(1:N) = SA * SX(1:N) + SB * SY(1:N).
BACKOP computes the number of back order terms for an ARIMA model.
BETAI computes the incomplete Beta ratio.
BFSDRV is the driver for time series Fourier spectrum analysis.
BFSER checks errors for time series Fourier univariate spectrum analysis.
BFS: short interface for time series Fourier bivariate spectrum analysis.
BFSF: short interface for time series Fourier bivariate spectrum analysis.
BFSFS: long interface for time series Fourier bivariate spectrum analysis.
BFSLAG: lag window truncation points for Fourier bivariate spectral analysis.
BFSM: short interface for time series bivariate Fourier spectrum analysis.
BFSMN computes square coherency and phase components of a bivariate spectrum.
BFSMS: long interface for BFS analysis with missing observations.
BFSMV: short interface for BFS analysis, missing observations, covariances.
BFSMVS: long interface for BFS analysis, missing observations, covariances.
BFSS: long call for time series bivariate Fourier spectrum analysis.
BFSV: short call for BFS analysis, with covariance input rather than series.
BFSVS: long call for BFS analsys with covariances input rather than series.
CCFER does error checking for CCF routines.
CCF computes the cross-correlation of two time series.
CCFF computes the cross-correlation of two time series by Singleton's FFT.
CCFFS computes multivariate cross-correlations and covariances by FFT.
CCFLST lists cross-correlations, standard errors, and summary information.
CCFM computes cross-correlation of two series with missing data.
CCFMN is the main routine for cross-correlations.
CCFMNF is the main routine for cross-correlations using an FFT.
CCFMNM is the main routine for cross-correlations with missing data.
CCFMS is a user routine for multivariate cross-correlations.
CCFOUT prints cross-correlations and standard errors.
CCFSD is the main routine for computing standard error of cross-correlations.
CCFSDM: standard error of cross-correlations with missing data.
CCFS is the user routine for multivariate cross-correlations.
CCFXP lists results for the time series cross-correlation routines.
CCVF computes the cross covariance function between two series.
CCVFF computes the cross covariance function between two series.
CCVFM computes the cross covariance function of two series with missing data.
CDFCHI computes the CDF for the Chi Square distribution.
CDFF computes the CDF for the F distribution.
CDFNML computes the CDF for the standard normal distribution.
CDFT computes the CDF for Student's T distribution.
CENTER centers an observed series.
CHIRHO computes the Chi Square statistic and its probability.
CMPFD computes a finite difference derivative.
CNTR centers the input seriers about its mean.
CORRER checks for errors in the input parameters.
CORR: short call to correlation family of routines.
CORRHD prints headers for the correlation family.
CORRMN is the main routine in the correlation family.
CORRS is the user routine for the correlation, with a long call interface.
CORRXP prints stored output returned from CORRS.
COVCLC computes the covariance matrix for NL2ITR.
CPYASF copies a symmetric matrix stored rowwise into rectangular storage.
CPYMSS copies an N by M matrix.
CPYVII copies an integer vector.
CSEVL evaluates a Chebyshev series.
D1MACH returns double precision machine constants.
DCKCNT controls the derivative checking process.
DCKCRV checks whether high curvature caused poor derivative approximation.
DCKDRV is the driver to the derivative checking routines.
DCKER does error checking for the derivative checking routines.
DCKFPA checks if arithmetic precision causes poor derivative approximation.
DCKHDR prints page headers for the derivative checking routines.
DCKLS1 sets up a problem for testing the step size selection family.
DCKLSC is the user routine for comparing analytic and numeric derivatives.
DCKLS is the user routine for comparing analytic and numeric derivatives.
DCKMN is the main routine for checking analytic versus numeric derivatives.
DCKOUT prints results from the derivative checking routine.
DCKZRO rechecks derivative errors where the analytic derivative is zero.
DCOEF expands a difference filter.
DEMDRV is the driver routine to demodulate a series.
DEMOD demodulates a series at a given frequency.
DEMODS demodulates a series at a given frequency.
DEMODU demodulates a series at a given frequency.
DEMORD sets up the data for the phase plots.
DEMOUT prints output for the time series demodulation routines.
DFAULT supplies default values to IV and V.
DFBW computes degrees of freedom and bandwidth for a given lag window.
DFBWM computes DOF and BW for a given lag window with missing data.
DIFC expands a difference filter and performs difference filtering.
DIF performs a first difference filtering operation.
DIFMC expands a difference filter and performs the difference filter.
DIFM performs a first difference filter for a series with missing data.
DIFSER performs a differencing operation on a series.
DOTC computes the dot product of two series, centered about their means.
DOTCM computes the dot product of series with missing data.
DOTPRD returns the inner product of two vectors.
DRV1A derivative function for NLS family exerciser subroutine MDL1.
DRV1B is an INCORRECT derivative function for the NLS exerciser MDL1.
DRV2 is a derivative function for the NLS exerciser routine MD12.
DRV3 is the derivative function for NLS family exerciser subroutine MDL3.
DRV4A is a (correct) derivative for testing derivative checking routines.
DRV4B is an (incorrect) derivative for testing derivative checking routines.
DRV is a dummy derivative function for the NLS family.
DUPDAT updates the scale vector for NL2ITR.
E9RINT stores the current error message or prints the old one.
ECVF prints an error message if missing data affects the covariance lags.
EHDR prints the heading for the error checking routines.
EIAGE ensures that "not too many" vectors are below a given lower bound.
EIAGEP prints the error messages for ERAGT and ERAGTM.
EISEQ prints an error message if NVAL is not equal to NEQ.
EISGE prints a warning if NVAL is less than NMIN.
EISII warns if an integer value does not lie within a given range.
EISLE warns if an integer is greater than a given maximum.
EISRNG warns if ISEED is not a suitable random number seed.
EIVEO checks whether all vector entries are even (or odd).
EIVEQ warns if the vector does not have at least NEQMN entries equal to IVAL.
EIVII warns if too many values are outside given limits.
ENFFT checks that NFFT is suitable for the Singleton FFT routine.
EPRINT prints the last error message, if any.
ERAGT warns if too many values are less than a lower bound.
ERAGTM warns if too many values are less than or equal to a lower bound.
ERAGTP prints the error messages for ERAGT and ERAGTM.
ERDF checks the values that specify differencing on a time series.
ERFC evaluates the complementary error function.
ERF evaluates the error function.
ERIODD warns if the value of NVAL is inconsistent.
ERSEI warns if a value is not between given limits.
ERSGE warns if a value is not greater than or equal to a minimum value.
ERSGT warns if the input value is not greater than a specified minumum.
ERSIE warns if a value is not within a specified range.
ERSII warns if the input value is not within the given range.
ERSLF checks the definition of a symmetric linear filter.
ERSLFS checks values specifying a symmetric linear filter for a time series.
ERVGT ensures that "most" values are greater than a specified lower bound.
ERVGTM ensures that "most" values are greater than a specified lower bound.
ERVGTP prints the error messages for ERVGT and ERVGTM.
ERVII checks for vector values outside given limits.
ERVWT checks user-supplied weights.
ETAMDL computes noise and number of good digits in model routine results.
EXTEND returns the I-th term in a series.
FACTOR factors an integer.
FDUMP is a dummy version of the dump routine called by XERRWV.
FFTCT does a cosine transform of n=2*n2 symmetric data points.
FFT is a multivariate complex Fourier transform.
FFTLEN computes the value of NFFT for the Singleton FFT routine.
FFTR is the user-callable routine for the Fourier transform of a series.
FITEXT checks whether the fit is exact to machine precision.
FITPT1 prints the data summary for nonlinear least squares routines.
FITPT2 prints the four standardized residual plots.
FITSXP generates reporst for least squares exerciser returned storage.
FITXSP generates reports for least squares exerciser returned storage.
FIXPRT sets the character array 'FIXED'.
FLTAR filters an input series using an autoregressive filter.
FLTARM filters a series with missing data, using an autoregressive filter.
FLTMA filters a series using a simple moving average filter.
FLTMD applies a modified Daniel filter to a symmetric series.
FTLSL filters an input series.
GAMI evaluates the incomplete Gamma function.
GAMIT evaluates Tricomi's incomplete Gamma function.
GAMLIM calculates the legal range of input arguments for the Gamma function.
GAMMA evaluates the Gamma function.
GAMR computes the reciprocal of the Gamma function.
GENI assigns an arithmetic sequence of values into an integer vector.
GENR puts an arithmetic sequence of values into a real vector.
GETPI returns the value of PI.
GFAEST computes the gain of an autoregressive linear filter.
GFARF: short call to compute gain function of autoregressive filter.
GFARFS: short call to compute gain function of autoregressive filter.
GFORD produces ordinants for the gain function plots.
GFOUT produces the gain function plots.
GFSEST: gain function of symmetric linear filter with given frequencies.
GFSLF: short call for gain function of symmetric linear filter.
GFSLFS: short call for gain function of symmetric linear filter.
GMEAN computes the geometric mean of a series.
GQTSTP computes the Goldfeld-Quandt-Trotter step by More-Hebden technique.
HIPASS carries out a high-pass filtering of a series.
HISTC: long call for producing a histogram.
HIST: short call for producing a histogram.
HPCOEF computes hi-pass filter given K-term low pass filter coefficients.
HPFLT compute high pass filter coefficients corresponding to a low pass filter.
HSTER does error checking for the HIST family of histogram routines.
HSTMN is the main routine for producing histograms.
I1MACH returns integer machine constants.
I8SAVE returns the current error number or recovery switch.
ICNTI counts the number of occurences of a value in an integer vector.
ICOPY copies an integer vector.
IMDCON returns integer machine-dependent constants.
INITS initializes an orthogonal series.
INPERL computes the number of elements that can be printed on one line.
IPGDV produces coordinates for the spectral plots.
IPGM: short call to compute the integrated periodogram of a series.
IPGMN computes the integrated periodogram.
IPGMP is the user routine for integrated periodograms of a series (short call).
IPGMPS is the user routine for the integrated periodogram of a series (long call).
IPGMS is the user routine fo the integrated periodogram of a series (long call).
IPGORD produces coordinates for the spectral plots.
IPGOUT produces the integrated periodogram plots.
IPRINT sets the logical unit for printed output.
ISAMAX indexes the real array element of maximum absolute value.
ITSMRY prints an iteration summary.
J4SAVE saves and recalls data needed by the XERROR error library.
LDSCMP computes storage needed for arrays.
LINVRT computes the inverse of a lower triangular matrix.
LITVMU solves L' * X = Y, where L is a lower triangular matrix.
LIVMUL solves L * X = Y, where L is a lower triangular matrix.
LLCNT is the controlling subroutine for linear least squares.
LLCNTG is the controlling subroutine for general linear least squares.
LLCNTP is the controlling subroutine for polynomial linear least squares.
LLER is the error checking routine for the linear least squares routines.
LLHDRG: page headings for the unrestricted linear least squares routines.
LLHDRP: page headings for polynomial linear least squares routines.
LLS is the general linear model least squares fit routine.
LLSMN: main program for the linear least squares fitting.
LLSP does an unweighted polynomial model least squares fit.
LLSPS does an unweighted polynomial model least squares fit.
LLSPW does a weighted polynomial model least squares fit.
LLSPWS computes a weighted polynomial model least squares fit.
LLSS computes an unweighted linear model least squares fit.
LLSW computes a weighted linear model least squares fit.
LLSWS performs a general linear model weighted least squares fit.
LMSTEP computes a Levenberg-Marquardt step by More-Hebden techniques.
LOGLMT adjusts plot limits for log plots, and computes log axis labels.
LOPASS carries out a low-pass filtering of a series.
LPCOEF computes a least squares approximation to an ideal low pass filter.
LPFLT computes the low-pass filter coefficients.
LSAME returns TRUE if CA is the same letter as CB regardless of case.
LSQRT computes the Cholesky factor of a lower triangular matrix.
LSTLAG finds the last computable lag value.
LSTVCF prints N elements of a masked array.
LSTVEC prints indices and values of a real vector.
LSVMIN estimates the smallest singular value of a lower triangular matrix.
LTSQAR sets A to the lower triangle of L' * L.
MADJ is a sample jacobian routine.
MADR is a sample residual routine.
MAFLT performs a moving average filtering operation.
MATPRF prints a square matrix stored in symmetric form.
MATPRT is a matrix printing routine.
MDFLT is a user routine for a modified Daniels filter of symmetric series.
MDL1 is the model function for an NLS exerciser.
MDL2 is a model function for an NLS exerciser.
MDL3 is a model function for an NLS exerciser.
MDL4 is a model routine for step size and derivative checking routines.
MDLTS1 is the user callable routine for estimating box-jenkins arima models.
MDLTS2 is the model routine for Pack's specification of box-jenkins models.
MDLTS3 is the user callable routine for estimating box-jenkins arima models.
MGS solves a linear system using modified Gram-Schmidt algorithm.
MODSUM prints the model summary for the ARIMA routines.
MPPC produces a simple page plot with multiple Y-axis values.
MPP produces a simple page plot with multiple Y-axis values.
MPPL produces a simple page plot with multiple Y-axis values, and log option.
MPPMC: produce a page plot with multiply Y-axis values, and missing data.
MPPM: produce a page plot with multiple Y-axis values and missing data.
MPPML: plot multiple Y-axis values with missing data, log option.
MSGX prints the returned and expected values for the error flag IERR.
MULTBP multiplies two difference factors from a Box-Jenkins time series model.
MVCHK checks whether the input value equals the flag value for missing data.
MVPC produces a vertical plot with multiple Y-axis values.
MVP produces a vertical plot with multiple Y-axis values.
MVPL produces a vertical plot with multiple y-axis values (log plot option).
MVPMC: vertical plot with missing data and multiple y-axis values (long call).
MVPM: vertical plot with missing data and multiple y-axis values (short call).
MVPML: vertical plot with missing data and multiple y-axis values (log option).
NCHOSE combines difference factors from a Box-Jenkins time series model.
NL2ITR carries out iterations for NL2SOL.
NL2SNO is like NL2SOL, but uses a finite difference jacobian.
NL2SOL minimizes a nonlinear sum of squares using an analytic jacobian.
NL2X tests nl2sol and nl2sno on madsen example.
NLCMP computes statistics for the NLS family when weights are involved.
NLCNTA: controlling routine for NLS regression with analytic derivatives.
NLCNT controlling subroutine for nonlinear least squares regression.
NLCNTN controlling routine for NLS regression with approximate derivatives.
NLDRVA computes the analytic derivative matrix from the user DERIV routine.
NLDRVN approximates the derivative matrix.
NLER does error checking routine for nonlinear least squares estimation.
NLERR sets the error flag ierr based on the convergence code returned by NL2.
NLFIN completes the NLS analysis once the estimates have been found.
NLHDRA prints headings for NLS estimation using analytic derivatives.
NLHDRN prints headings for NLS estimation using approximate derivatives.
NLINIT initializes the NLS routines.
NLISM prints an initial summary for the nonlinear least squares routines.
NLITRP prints iteration reports for nonlinear least squares regression.
NLMN: controlling routine for nonlinear least squares regression.
NLOUT prints the final summary report for nonlinear least squares routines.
NLSC: NLS regression, approximate derivatives (control call)
NLSDC: NLS regression, analytic derivatives, user parameters.
NLSD: nonlinear least squares regression, analytic derivatives (short call).
NLSDS: NLS regression, analytic derivatives, user parameters.
NLS: NLS regression with numeric derivatives, short call.
NLSKL prints warning messages for the nonlinear least squares routines.
NLSPK packs the unmasked elements of one vector into another.
NLSS: interface for nonlinear least squares reqression, approximate derivatives.
NLSUPK unpacks a vector into another, using a mask vector.
NLSWC: nonlinear least squares regression with weights and approximate derivatives.
NLSWDC: NLS regression, analytic derivatives, weights, user parameters.
NLSWD: NLS regression, analytic derivatives, weights (short call).
NLSWDS: NLS regression with analytic derivatives, weights, user parameters.
NLSW: NLS regression with approximate derivatives and weights.
NLSWS: NLS regression with approximate derivatives and weights.
NLSX1 sets the starting parameter values for NLSX.
NLSX2 sets a problem for testing the NLS family.
NRANDC generates pseudorandom normally distributed values.
NRAND generates pseudorandom normally distributed values.
OANOVA computes and prints analysis of variance.
OBSSM2 lists the data summary for the arima estimation routines.
OBSSUM lists the data summary for the least squares subroutines.
PARCHK checks the NL2SOL parameters.
PARZEN computes and stores the Parzen lag window.
PGMEST computes the periodogram estimates.
PGM is the user callable routine for the raw periodogram of a series.
PGMMN is the main routine for computing the raw periodogram.
PGMS computes the (raw) periodogram of a series (long call).
PGORD produces coordinates for the periodogram plot.
PGOUT produces the periodogram plots.
PLINE defines one line of a plot string for the vertical plot routines.
PLTCHK checks for errors for the multiple plot routines.
PLTPLX computes the point location in the plot string.
PLTSYM supplies the plot symbol for the plot line.
POLAR converts complex numbers from Cartesian to polar representation.
PPC produces a simple page plot (long call).
PPCNT is the controling routine for user called page plot routines.
PP is the user callable routine which produces a simple page plot (short call).
PPCHFS computes the percentage points of the Chi Square distribution.
PPFF computes the percentage points for the F distribution.
PPFNML computes the percentage points of the normal distribution.
PPFT computes the percentage points of the Student's T distribution.
PPL produces a simple page plot (log option).
PPLMT sets the plot limits for page plots with missing values.
PPMC produces a simple page plot for data with missing observations (long call).
PPM produces a simple page plot for data with missing observations (short call).
PPML plots data with missing observations (log option).
PPMN is the main routine for page plots.
PRTCNT sets up the print control parameters.
QAPPLY applies orthogonal transformation to the residual R.
QRFACT computes the QR decomposition of a matrix.
R1MACH returns single precision machine constants.
R9GMIT computes Tricomi's incomplete gamma function for small X.
R9LGIC compute the log complementary incomplete gamma function.
R9LGIT computes the log of Tricomi's incomplete Gamma function.
R9LGMC computes the log Gamma correction factor.
RANDN returns normal random numbers.
RANDU returns uniform random numbers.
RANKO puts the rank of N X's in the vector R.
REALTR computes the forward or inverse Fourier transform of real data.
RELCOM computes the difference between V(I) and W(I) relative to RELTOL.
RELDST computes the relative difference between two real values.
REPCK reformats the data in D for the N by NPAR format used by NLCMP.
RMDCON returns machine constants.
RPTMUL multiplies the R factor times a vector X.
S88FMT writes an integer into a string.
SAMPLE creates a new series by sampling every K-th item of the input.
SASUM takes the sum of the absolute values of a real vector.
SAXPY adds a real constant times one vector to another.
SCOPY copies one real vector into another.
SDOT forms the dot product of two real vectors.
SETERR sets the error number and prints the error message.
SETESL computes the smallest suitable value of NFFT for given N and Singleton FFT.
SETFRQ computes the frequencies at which the spectrum is to be estimated.
SETIV sets the entries of an integer vector to a value.
SETLAG sets the number of autocorrelations to be computed.
SETRA sets the entries of a real array to a given value.
SETROW selects the row used by the derivative checking procedure.
SETRV sets the elements of a real vector to a value.
SLFLT applies a symmetric filter to a series.
SLUPDT updates a symmetric matrix A so that A * STEP = Y.
SLVMUL sets Y = S * X, where S is a P by P symmetric matrix.
SMACH computes machine parameters for single precision arithmetic.
SMPLY samples every K-th observation from a series.
SNRM2 computes the Euclidean norm of a real vector.
SPCCK analyzes ordinates for the spectal semi-log plots.
SPPC produces a simple page plot with user control of plot symbols (long call).
SPP produces a simple page plot with user control of plot symbols (short call).
SPPL produces a simple page plot with user control of plot symbols (log option).
SPPLTC: confidence interval and bandwidth coordinates for spectrum plots.
SPTLTD sets various y axis limits for decibel spectrum plots.
SPPLTL sets various y axis limits for decibel spectrum plots.
SPPMC: page plot with user plot symbols and missing observations (long call).
SPPM page plot with user plot symbols and missing observations (short call).
SPPML: page plot with user plot symbols and missing observations (log option).
SROT applies a real plane rotation.
SROTG constructs a real Givens plane rotation.
SRTIR sorts an integer array IR on a key array A.
SRTIRR sorts arrays A, IR and RR based on values in A.
SRTRI sorts array A on integer array IR.
SRTRRI sorts the array IR, A and RR, based on values in IR.
SSCAL scales a real vector by a constant.
SSIDI computes the determinant, inertia and inverse of a real symmetric matrix.
SSIFA factors a real symmetric matrix.
SSWAP interchanges two real vectors.
STAT1 computes statistics for a sorted vector.
STAT1W computes statistics for a sorted vector with weights.
STAT2 computes statistics that do not require sorted data.
STAT2W computes statistics on an unsorted, weighted vector.
STATER does error checking for the STAT family of routines.
STAT computes 53 statistics for an unweighted vector.
STATS computes 53 different statistics for an unweighted vector.
STATW computes 53 statistics for a weighted vector.
STATWS computes 53 statistics for a weighted vector.
STKCLR clears the stack for framework area manipulation routines.
STKGET allocates space on an integer stack.
STKREL de-allocates the last allocations made in the stack.
STKSET initializes the stack to NITMES of type ITYPE.
STKST returns statistics on the state of the stack.
STOPX is called to stop execution.
STPADJ adjusts the selected step sizes to optimal values.
STPAMO is a dummy routine for the arima estimation routines.
STPCNT controls the stepsize selection process.
STPDRV is the driver for selecting forward difference step sizes.
STPER does error checking for the stepsize selection routines.
STPHDR prints page headings for the stepsize selection routines.
STPLS1 sets a test problem for the step size selection family.
STPLS2 sets a test problem for the step size selection family.
STPLSC selects step sizes for forward difference estimates of derivatives.
STPLS selects step sizes for estimating derivatives in NLS routines.
STPMN: main routine for numerical derivative step size selection.
STPOUT prints results for the step size selection routines.
STPSEL selects new step sizes until no further improvement can be made.
STRCO estimates the condition of a real triangular matrix.
STRDI computes the determinant and inverse of a real triangular matrix.
SUMBS finds a zero or value closest to zero in a sorted vector.
SUMDS sums unweighted powers of differences from the mean of a sorted vector.
SUMID sums I * ( X(I) - XMEAN ).
SUMIDW: dot product of I and ( X(I) - XMEANW ).
SUMOT reports the computation of 53 selected statistics.
SUMSS calculates the sum of powers and mean for a sorted vector.
SUMTS calculates unweighted trimmed mean for a sorted vector.
SUMWDS calculates sums of powers of differences from the weighted mean.
SUMWSS calculates weighted and unweighted sums of powers and the mean.
SUMWTS calculates weighted and unweighted means for a sorted vector.
SVPC produces a vertical plot with user plot symbols (long call).
SVP: vertical plot with user plot symbols (short call).
SVPL produces a vertical log plot with user control of the plot symbol.
SVPMC: vertical plot with missing data and user plot symbols (long call).
SVPM: vertical plot with missing data and user plot symbols (short call).
SVPML: vertical plot with missing data and user plot symbols (log plot option).
TAPER applies a split-cosine-bell taper to a centered observed series.
TIMESTAMP prints the current YMDHMS date as a time stamp.
UAS is the user callable routine for autoregressive spectrum estimation.
UASCFT computes autoregressive coefficients using Durbin's method.
UASDV is the driver for computing the autoregressive and Fourier spectrums.
UASER: error checks for time series Fourier univariate spectrum analysis.
UASET calculates the autoregressive spectrum.
UASF is the user callable routine for autoregressive spectrum estimation.
UASFS: interface for autoregressive spectrum estimation using the FFT (long call).
UASORD produces coordinates for the spectrum plots.
UASOUT produces the spectrum plots for the autoregressive spectrum estimates.
UASS: user interface for autoregressive spectrum estimation (long call).
UASVAR computes the variance for a given series and autoregressive model.
UASV is the user routine for autoregressive spectrum estimation.
UASVS is a user routine for autoregressive spectrum estimation.
UFPARM is a dummy version of the optional user function for NL2SOL.
UFSDRV is the controlling routine for time series Fourier spectrum analysis.
UFSET checks errors for time series Fourier univariate spectrum analysis.
UFSEST computes the spectra and the confidence limits.
UFS: user routine for time series Fourier spectrum analysis (short call).
UFSF: user routine for Fourier spectrum analysis using fft (short call).
UFSFS: user routine for Fourier spectrum analysis using the fft (long call).
UFSLAG computes the lag window truncation points for spectrum analysis.
UFSM: user routine for Fourier spectrum analysis with missing data (short call).
UFSMN computes autocorrelations and partial autocorrelations of a time series.
UFSMS: time series Fourier spectrum analysis with missing data (long call).
UFSMV: Fourier spectrum analysis, missing data, user ACVF values (short call).
UFSMVS: time series Fourier spectrum analysis with missing data (long call).
UFSOUT produces the Fourier bivariate spectrum output.
UFSPCV produces coordinates for the spectrum plots.
UFSS: time series Fourier spectrum analysis (long call).
UFSV: Fourier spectrum analysis, user supplied ACVF values (short call).
UFSVS: Fourier spectrum analysis and user supplied acvf values (long call).
V2NORM computes the L2 norm of a vector.
VCOPY copies a vector.
VCVOTF prints the variance-covariance matrix.
VCVOUT prints the variance-covariance matrix.
VERSP prints the version number.
VP is the user callable routine which produces a vertical plot (short call).
VPC is the user callable routine which produces a vertical plot (long call).
VPCNT is the controlling routine for user-called vertical plots
VPHEAD prints the heading for the vertical plot output.
VPL is the user callable routine which produces a vertical log plot.
VPLMT sets the plot limits for vertical plots
VPM produces a vertical plot with missing data (short call).
VPMC produces a vertical plot with missing data (long call).
VPML produces a vertical plot with missing data (log plot option).
VPMN produces vertical plots.
XERABT aborts execution of the program.
XERBLA is an error handler for the LAPACK routines.
XERCLR resets the current error number to zero.
XERCTL gives the user control over handling individual errors.
XERPRT prints an error message.
XERROR processes a diagnostic message.
XERRWV processes a diagnostic message.
XERSAV records that an error has occurred.
XGETF returns the current error control flag.
XGETUA determines the units to which error messages are being sent.
XSETF sets the error control flag.
```
