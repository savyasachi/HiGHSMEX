/* MATLAB mex wrapper over the irreducible infeasible subsystem (IIS) facility of
 * the HiGHS optimization library (https://github.com/ERGO-Code/HiGHS)
 *
 * This is a stand-alone companion to highsmex.cpp. It builds an LP from the same
 * leading input arguments as highsmex (c, A, L, U, l, u) and calls
 * Highs::getIis(HighsIis&) instead of Highs::run().
 *
 * The HiGHS IIS facility is documented at
 * https://ergo-code.github.io/HiGHS/stable/guide/advanced/
 * and is currently available for LPs only (no MIP, no QP).
 *
 * Covered by the MIT License (see LICENSE file for details).
 * See https://github.com/savyasachi/HiGHSMEX for more information.
 */


/* Include the code shared with highsmex. It also pulls in the standard headers,
 * the C++ MEX API and HiGHS. Highs.h in turn pulls in lp_data/HighsIis.h, which
 * defines HighsIis, HighsIisInfo, IisBoundStatus and IisModelStatus. The
 * IisStrategy and IisStatus enumerations come from lp_data/HConst.h.
 */
#include "highsmex_common.hpp"
// The mex entry point. It must appear in exactly one translation unit per mex file.
#include "mexAdapter.hpp"


/* ------------------------------------------------------------------------------------------------------ */
/*                                        ENUMERATIONS                                                    */
/* ------------------------------------------------------------------------------------------------------ */

enum class MexCallSyntax { kVer, kDefaultOpts, kIntType, kIis };


/* ------------------------------------------------------------------------------------------------------ */
/*                                         VARIABLES                                                      */
/* ------------------------------------------------------------------------------------------------------ */

/* Default value used for the HiGHS option "iis_strategy".
 * The HiGHS default is kIisStrategyLight (0), which only performs the cheap
 * trivial checks (inconsistent bounds, empty infeasible rows, row value bounds)
 * and usually returns an empty IIS. Since the whole point of this mex function
 * is to determine an IIS, the default here is
 *   kIisStrategyFromLp (2)      - build a mutually infeasible set of rows using
 *                                 the elasticity filter, and
 *   kIisStrategyIrreducible (4) - reduce that set to an irreducible one.
 * Pass an "iis_strategy" field in the options struct to override this.
 */
constexpr HighsInt kHighsMexDefaultIisStrategy = kIisStrategyFromLp | kIisStrategyIrreducible;

// Maps of the HiGHS IIS enumerations onto the strings reported to MATLAB
const std::map<HighsInt, std::string> iisBoundStatusToStringMap({
	{kIisBoundStatusDropped, "dropped"},
	{kIisBoundStatusNull,    "null"},
	{kIisBoundStatusFree,    "free"},
	{kIisBoundStatusLower,   "lower"},
	{kIisBoundStatusUpper,   "upper"},
	{kIisBoundStatusBoxed,   "boxed"}
	});
const std::map<HighsInt, std::string> iisStatusToStringMap({
	{kIisStatusNotInConflict,   "notInConflict"},
	{kIisStatusMaybeInConflict, "maybeInConflict"},
	{kIisStatusInConflict,      "inConflict"}
	});
const std::map<HighsInt, std::string> iisModelStatusToStringMap({
	{kIisModelStatusFeasible,    "feasible"},
	{kIisModelStatusUnknown,     "unknown"},
	{kIisModelStatusTimeLimit,   "timeLimit"},
	{kIisModelStatusReducible,   "reducible"},
	{kIisModelStatusIrreducible, "irreducible"}
	});


/* ------------------------------------------------------------------------------------------------------ */
/*                                      IIS CONVERSIONS                                                   */
/* ------------------------------------------------------------------------------------------------------ */


/* Convert a vector of zero based HiGHS indices to a MATLAB (one based) column
 * vector of doubles, so that the result can be used for indexing straight away.
 */
TypedArray<double> highsIndexVectorToMatlabVector(const std::vector<HighsInt>& v) {
	auto out = factory.createArray<double>(ArrayDimensions({ v.size(), 1 }));
	auto p = getPointer(out);
	for (size_t i = 0; i < v.size(); ++i) {
		p[i] = static_cast<double>(v[i]) + 1; // +1 to convert C++ (zero) based index to MATLAB (one) based index
	}
	return out;
}


// Convert a vector of HiGHS enumeration values to a MATLAB column vector of strings.
TypedArray<MATLABString> highsIntVectorToMatlabStringVector(const std::vector<HighsInt>& v, const std::map<HighsInt, std::string>& valueToStringMap) {
	auto out = factory.createArray<MATLABString>(ArrayDimensions({ v.size(), 1 }));
	std::transform(v.begin(), v.end(), out.begin(),
		[&valueToStringMap](const HighsInt value_) -> std::string {
			auto const it_ = valueToStringMap.find(value_);
			return it_ == valueToStringMap.end() ? std::string("unknown") : it_->second;
		});
	return out;
}


// Look up an enumeration value in one of the maps above, without throwing for unknown values.
std::string highsIntToString(const HighsInt value, const std::map<HighsInt, std::string>& valueToStringMap) {
	auto const it = valueToStringMap.find(value);
	return it == valueToStringMap.end() ? std::string("unknown") : it->second;
}


/* Convert a HighsSparseMatrix to a MATLAB cell array in the {i, j, v, nr, nc}
 * format accepted by callhighs, where [i, j, v]=find(A) and [nr, nc]=size(A).
 * The row and column indices are one based. Both the col-wise and the row-wise
 * HiGHS matrix formats are handled.
 */
CellArray highsSparseMatrixToMatlabCell(const HighsSparseMatrix& matrix) {
	const HighsInt numRow = matrix.num_row_;
	const HighsInt numCol = matrix.num_col_;
	const bool rowwise = matrix.format_ == MatrixFormat::kRowwise;
	// A matrix with no entries may carry an empty start_ vector, in which case
	// there is nothing to iterate over.
	const HighsInt numVec = matrix.start_.empty()
		? 0
		: std::min(rowwise ? numRow : numCol, castToHighsInt(matrix.start_.size()) - 1);
	const size_t nnz = numVec > 0 ? static_cast<size_t>(matrix.start_[numVec]) : 0;

	auto iArr = factory.createArray<double>(ArrayDimensions({ nnz, 1 }));
	auto jArr = factory.createArray<double>(ArrayDimensions({ nnz, 1 }));
	auto vArr = factory.createArray<double>(ArrayDimensions({ nnz, 1 }));
	if (nnz) {
		auto pI = getPointer(iArr), pJ = getPointer(jArr), pV = getPointer(vArr);
		size_t k = 0;
		for (HighsInt iVec = 0; iVec < numVec; ++iVec) {
			for (HighsInt iEl = matrix.start_[iVec]; iEl < matrix.start_[iVec + 1]; ++iEl) {
				const HighsInt indx = matrix.index_[iEl];
				// +1 to convert C++ (zero) based indices to MATLAB (one) based indices
				pI[k] = static_cast<double>(rowwise ? iVec : indx) + 1;
				pJ[k] = static_cast<double>(rowwise ? indx : iVec) + 1;
				pV[k] = matrix.value_[iEl];
				++k;
			}
		}
	}
	return factory.createCellArray({ 1, 5 },
		std::move(iArr), std::move(jArr), std::move(vArr),
		factory.createScalar<double>(static_cast<double>(numRow)),
		factory.createScalar<double>(static_cast<double>(numCol)));
}


// Convert HighsIisInfo to a MATLAB struct.
StructArray highsIisInfoToMatlabStruct(const HighsIisInfo& info) {
	auto out = factory.createStructArray({ 1, 1 },
		{ "num_lp_solved", "sum_simplex_iteration_counts", "min_simplex_iteration_count",
		  "max_simplex_iteration_count", "sum_simplex_times", "min_simplex_time", "max_simplex_time" });
	out[0]["num_lp_solved"] = factory.createScalar(info.num_lp_solved);
	out[0]["sum_simplex_iteration_counts"] = factory.createScalar(info.sum_simplex_iteration_counts);
	out[0]["min_simplex_iteration_count"] = factory.createScalar(info.min_simplex_iteration_count);
	out[0]["max_simplex_iteration_count"] = factory.createScalar(info.max_simplex_iteration_count);
	out[0]["sum_simplex_times"] = factory.createScalar(info.sum_simplex_times);
	out[0]["min_simplex_time"] = factory.createScalar(info.min_simplex_time);
	out[0]["max_simplex_time"] = factory.createScalar(info.max_simplex_time);
	return out;
}


/* Convert HighsIis to a MATLAB struct. The fields mirror the data members of the
 * HighsIis class, with the indices converted to MATLAB (one based) indices and
 * the enumerations converted to strings.
 */
StructArray highsIisToMatlabStruct(const HighsIis& iis) {
	auto out = factory.createStructArray({ 1, 1 },
		{ // These fields mirror the data members of the HighsIis class of the HiGHS library
			"valid", "status", "strategy", "col_index", "row_index", "col_bound", "row_bound",
			"col_status", "row_status", "info",
			// These fields are extra. They are added by highsmex_iis.
			"status_string" });
	out[0]["valid"] = factory.createScalar(iis.valid_);
	out[0]["status"] = factory.createScalar(iis.status_);
	out[0]["strategy"] = factory.createScalar(iis.strategy_);
	out[0]["col_index"] = highsIndexVectorToMatlabVector(iis.col_index_);
	out[0]["row_index"] = highsIndexVectorToMatlabVector(iis.row_index_);
	out[0]["col_bound"] = highsIntVectorToMatlabStringVector(iis.col_bound_, iisBoundStatusToStringMap);
	out[0]["row_bound"] = highsIntVectorToMatlabStringVector(iis.row_bound_, iisBoundStatusToStringMap);
	out[0]["col_status"] = highsIntVectorToMatlabStringVector(iis.col_status_, iisStatusToStringMap);
	out[0]["row_status"] = highsIntVectorToMatlabStringVector(iis.row_status_, iisStatusToStringMap);
	out[0]["info"] = highsIisInfoToMatlabStruct(iis.info_);
	// Extra fields
	out[0]["status_string"] = factory.createScalar(highsIntToString(iis.status_, iisModelStatusToStringMap));
	return out;
}


/* Convert the LP held by a HighsIis (the infeasible subsystem itself) to a MATLAB
 * struct laid out like the input arguments of callhighs, so that it can be passed
 * straight back to callhighs / callhighs_iis.
 *
 * NOTE: HiGHS sets the column costs of the IIS LP to zero, since costs play no
 *       role in infeasibility. The bounds are the ones that characterise the IIS,
 *       i.e. bounds that were dropped during the reduction are infinite here.
 */
StructArray highsIisLpToMatlabStruct(const HighsIis& iis) {
	auto const& lp = iis.model_.lp_;
	auto out = factory.createStructArray({ 1, 1 },
		{ "c", "A", "L", "U", "l", "u", "colIndex", "rowIndex" });
	out[0]["c"] = stdVectorToMatlabVector(lp.col_cost_, false);
	out[0]["A"] = highsSparseMatrixToMatlabCell(lp.a_matrix_);
	out[0]["L"] = stdVectorToMatlabVector(lp.row_lower_, false);
	out[0]["U"] = stdVectorToMatlabVector(lp.row_upper_, false);
	out[0]["l"] = stdVectorToMatlabVector(lp.col_lower_, false);
	out[0]["u"] = stdVectorToMatlabVector(lp.col_upper_, false);
	out[0]["colIndex"] = highsIndexVectorToMatlabVector(iis.col_index_);
	out[0]["rowIndex"] = highsIndexVectorToMatlabVector(iis.row_index_);
	return out;
}


/* ------------------------------------------------------------------------------------------------------ */
/*                                         MEX INTERFACE                                                  */
/* ------------------------------------------------------------------------------------------------------ */

class MexFunction : public HighsMexBase {

	MexCallSyntax checkMexCallSyntax(ArgumentList& inputs) {
		if (inputs.size() == 1) {
			if (!(isMatlabString(inputs[0]) && isScalar(inputs[0]))) {
				throw std::runtime_error("Input argument must be a MATLAB string.");
			}
			const TypedArray<MATLABString> in0(inputs[0]);
			const std::string instr = matlabStringToStdString(in0[0]);
			if (instr == "ver") {
				return MexCallSyntax::kVer;
			}
			else if (instr == "defopts") {
				return MexCallSyntax::kDefaultOpts;
			}
			else if (instr == "intType") {
				return MexCallSyntax::kIntType;
			}
			else {
				throw std::runtime_error("Input string is invalid.");
			}
		}
		else if (inputs.size() >= 2 && inputs.size() <= 7) {
			return MexCallSyntax::kIis;
		}
		else {
			throw std::runtime_error("Invalid number of input arguments.");
		}
	}

	/* Determine the number of columns implied by the second input argument (A).
	 * Returns false if A is omitted (empty), in which case numCol is not set.
	 */
	bool numColFromMatrixArgIn(ArgumentList& inputs, HighsInt& numCol) {
		if (isEmpty(inputs[1])) return false;
		if (isDouble(inputs[1])) {
			auto const dims = inputs[1].getDimensions();
			if (!isMatrix(dims)) {
				throw std::runtime_error("Second input argument (A) must be a matrix of double type.");
			}
			numCol = castToHighsInt(dims[1]);
			return true;
		}
		if (isCell(inputs[1])) {
			if (numel(inputs[1]) != 5) throw std::runtime_error("Second input argument (A) must be a cell array of 5 elements.");
			const CellArray cell(inputs[1]);
			if (!(isDouble(cell[4]) && isScalar(cell[4]))) {
				throw std::runtime_error("The 5th element of the cell array passed as the second input argument (A) must be a double scalar representing the number of columns of A.");
			}
			const TypedArray<double> ncolsA = cell[4];
			numCol = castToHighsInt(ncolsA[0]);
			return true;
		}
		throw std::runtime_error("Second input argument (A) must be a matrix of double type or, a cell array of 5 elements.");
	}

	/* Process the first input argument (c) and set the number of columns of the LP.
	 * Unlike callhighs, c may be empty, in which case the number of columns is taken
	 * from A and all the column costs are set to zero. Column costs play no role in
	 * the IIS calculation, they are accepted only so that the leading input arguments
	 * match those of callhighs.
	 */
	void process1stArgIn(ArgumentList& inputs, HighsLp& lp) {
		HighsInt numColA = 0;
		const bool haveA = numColFromMatrixArgIn(inputs, numColA);

		if (isEmpty(inputs[0])) {
			if (!haveA) {
				throw std::runtime_error("First input argument (c) may only be empty when the second input argument (A) is not empty.");
			}
			lp.num_col_ = numColA;
			lp.col_cost_.assign(lp.num_col_, 0);
		}
		else {
			auto const dims = inputs[0].getDimensions();
			if (!(isDouble(inputs[0]) && isVector(dims))) {
				throw std::runtime_error("First input argument (c) must be a double type vector, or an empty array.");
			}
			const TypedArray<double> c(inputs[0]);
			lp.num_col_ = castToHighsInt(numel(c));
			lp.col_cost_ = matlabVectorToStdVector(c);
			if (haveA && numColA != lp.num_col_) {
				throw std::runtime_error(formatMessage("The second input argument (A) must have ", lp.num_col_, " columns to match the length of the first input argument (c)."));
			}
		}
		lp.offset_ = 0;

		if constexpr (MexDebugPrinting) {
			print("lp.col_cost_ = "); disp(lp.col_cost_);
		}
	}

	// Pre-condition: lp.num_col_ must be set.
	void process2ndArgIn(ArgumentList& inputs, HighsLp& lp) {
		if (isEmpty(inputs[1])) { // No linear constraints, hence no matrix A
			lp.num_row_ = 0;
			lp.a_matrix_.format_ = MatrixFormat::kColwise;
			lp.a_matrix_.start_.assign(lp.num_col_ + 1, 0);
			lp.a_matrix_.num_col_ = lp.num_col_;
			lp.a_matrix_.num_row_ = 0;
			return;
		}

		bool isSparse = false;
		if (isDouble(inputs[1])) {
			auto const dims = inputs[1].getDimensions();
			// The column count was already checked against c in process1stArgIn
			lp.num_row_ = castToHighsInt(dims[0]);
		}
		else { // Cell array. Already validated in numColFromMatrixArgIn.
			isSparse = true;
			const CellArray cell(inputs[1]);
			if (!(isDouble(cell[3]) && isScalar(cell[3]))) {
				throw std::runtime_error("The 4th element of the cell array passed as the second input argument (A) must be a double scalar representing the number of rows of A.");
			}
			const TypedArray<double> nrowsA = cell[3];
			lp.num_row_ = castToHighsInt(nrowsA[0]);
		}

		lp.a_matrix_.format_ = MatrixFormat::kColwise;
		if (isSparse) {
			const CellArray cell(inputs[1]);
			auto isDoubleVec = [](const Array& arr_) -> bool { return isDouble(arr_) && (isEmpty(arr_) || isVectorArr(arr_)); };
			if (!(isDoubleVec(cell[0]) && isDoubleVec(cell[1]) && isDoubleVec(cell[2]))) {
				throw std::runtime_error("The 1st, 2nd, and 3rd elements of the cell array passed as the second input argument (A) must be double type vectors.");
			}
			const TypedArray<double> iA = cell[0];
			const TypedArray<double> jA = cell[1];
			const TypedArray<double> vA = cell[2];
			auto const nA = numel(iA);
			if (!(nA == numel(jA) && nA == numel(vA))) {
				throw std::runtime_error("The 1st, 2nd, and 3rd elements of the cell array passed as the second input argument (A) must be vectors of the same length.");
			}
			matlabMatrixToHighsFormat(lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_,
				iA, jA, vA, lp.num_row_, lp.num_col_);
		}
		else {
			const TypedArray<double> A(inputs[1]);
			matlabMatrixToHighsFormat(lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_,
				A, lp.num_row_, lp.num_col_);
		}
		lp.a_matrix_.num_col_ = lp.num_col_;
		lp.a_matrix_.num_row_ = lp.num_row_;

		if constexpr (MexDebugPrinting) {
			print("lp.a_matrix_.start_ = "); disp(lp.a_matrix_.start_);
			print("lp.a_matrix_.index_ = "); disp(lp.a_matrix_.index_);
			print("lp.a_matrix_.value_ = "); disp(lp.a_matrix_.value_);
		}
	}

	/* Process one of the four bound vectors (L, U, l, u).
	 * Pre-condition: numExpected is the number of rows (for L and U) or the number
	 * of columns (for l and u) of the LP.
	 */
	void processBoundArgIn(ArgumentList& inputs, std::vector<double>& bound, const size_t argIndx,
		const HighsInt numExpected, const double defaultValue, const std::string& argDescription) {
		if (!numExpected) { // Nothing to bound, e.g. L and U when there are no linear constraints
			if (argIndx < inputs.size() && !isEmpty(inputs[argIndx])) {
				throw std::runtime_error(formatMessage("Expected ", argDescription, " to be an empty array."));
			}
			return;
		}

		bool setToDefault = true;
		if (argIndx < inputs.size() && !isEmpty(inputs[argIndx])) {
			auto const dims = inputs[argIndx].getDimensions();
			if (!(isDouble(inputs[argIndx]) && isVector(dims))) {
				throw std::runtime_error(formatMessage(argDescription, " must be a vector of double type, or an empty array."));
			}
			setToDefault = false;
		}

		if (setToDefault) {
			bound.assign(numExpected, defaultValue);
		}
		else {
			const TypedArray<double> b(inputs[argIndx]);
			if (castToHighsInt(numel(b)) != numExpected) {
				throw std::runtime_error(formatMessage("Expected length of ", argDescription, " to be ", numExpected, "."));
			}
			bound = matlabVectorToStdVector(b);
		}

		if constexpr (MexDebugPrinting) {
			print(formatMessage(argDescription, " = ")); disp(bound);
		}
	}

	// Process the seventh input argument (options).
	void processOptionsArgIn(ArgumentList& inputs, Highs& highs) {
		if (inputs.size() <= 6 || isEmpty(inputs[6])) return; // Use the default options
		if (!(isStruct(inputs[6]) && isScalar(inputs[6]))) {
			throw std::runtime_error("Seventh input argument (options) must be a 1x1 MATLAB struct, or an empty array.");
		}
		const StructArray opts(inputs[6]);
		setHighsOptions(highs, opts, "seventh");
	}

	void runIis(ArgumentList& inputs, Highs& highs, HighsIis& iis) {
		// Build the LP from the input arguments
		HighsLp lp;
		process1stArgIn(inputs, lp);
		process2ndArgIn(inputs, lp);
		processBoundArgIn(inputs, lp.row_lower_, 2, lp.num_row_, -kHighsInf, "third input argument (L)");
		processBoundArgIn(inputs, lp.row_upper_, 3, lp.num_row_, kHighsInf, "fourth input argument (U)");
		processBoundArgIn(inputs, lp.col_lower_, 4, lp.num_col_, -kHighsInf, "fifth input argument (l)");
		processBoundArgIn(inputs, lp.col_upper_, 5, lp.num_col_, kHighsInf, "sixth input argument (u)");

		/* Ask for a full IIS calculation by default. This is done before applying the
		 * user supplied options so that the user can override it.
		 */
		if (highs.setOptionValue("iis_strategy", kHighsMexDefaultIisStrategy) != HighsStatus::kOk) {
			throw std::runtime_error("Failed to set the HiGHS option \"iis_strategy\".");
		}

		// Set the user supplied options
		processOptionsArgIn(inputs, highs);

		bool log_to_console;
		highs.getOptionValue("log_to_console", log_to_console);
		if (log_to_console) {
			// Set callback with Highs
			auto callback = [this](int callbackType, const std::string& message, const HighsCallbackOutput* dataOut, HighsCallbackInput*, void*) -> void {
				logCallback(callbackType, static_cast<HighsLogType>(dataOut->log_type), message);
				};
			if (highs.setCallback(callback, nullptr) != HighsStatus::kOk) {
				throw std::runtime_error("Failed to set the logging callback with HiGHS.");
			}
			checkHighsReturnStatus(highs.startCallback(HighsCallbackType::kCallbackLogging),
				"Warning issued when attempting to start the logging callback.",
				"Failed to start the logging callback.");
			highs.setOptionValue("log_to_console", false);
		}

		// Pass the LP to HiGHS
		checkHighsReturnStatus(highs.passModel(lp),
			"Warning issued when passing the model to the HiGHS solver.",
			"Failed to pass the model to the HiGHS solver.");

		/* Compute the IIS. Highs::getIis solves the model first if its status is not
		 * yet known, so there is no need to call Highs::run beforehand.
		 */
		checkHighsReturnStatus(highs.getIis(iis),
			"Warning issued when computing the irreducible infeasible subsystem (IIS). The returned subsystem may not be valid, check the \"valid\" field of the first output argument.",
			"Failed to compute the irreducible infeasible subsystem (IIS).");
	}

public:

	/* This is the gateway routine for the MEX-file. */
	void operator()(ArgumentList outputs, ArgumentList inputs) {
		try {
			Highs highs;
			HighsIis iis;

			switch (checkMexCallSyntax(inputs)) {
			case MexCallSyntax::kVer:
				if (outputs.size() != 1) throw std::runtime_error("Number of output arguments must be one.");
				outputs[0] = factory.createScalar(highs.version());
				return;

			case MexCallSyntax::kDefaultOpts:
				if (outputs.size() != 1) throw std::runtime_error("Number of output arguments must be one.");
				outputs[0] = highsOptionsToMatlabStruct(highs, true);
				return;

			case MexCallSyntax::kIntType:
				if (outputs.size() != 1) throw std::runtime_error("Number of output arguments must be one.");
				outputs[0] = factory.createScalar(HighsInt2MatlabClassStr);
				return;

			case MexCallSyntax::kIis:
				if (!(outputs.size() >= 1 && outputs.size() <= 4)) {
					throw std::runtime_error("Number of output arguments must be >= 1 and <= 4.");
				}
				// Clear the error stack
				while (!highsLogErrStack.empty()) {
					highsLogErrStack.pop();
				}
				// Process inputs and compute the IIS
				runIis(inputs, highs, iis);
				// Assign outputs
				outputs[0] = highsIisToMatlabStruct(iis);
				if (outputs.size() > 1) {
					outputs[1] = highsIisLpToMatlabStruct(iis);
					if (outputs.size() > 2) {
						outputs[2] = highsInfoToMatlabStruct(highs);
						if (outputs.size() > 3) {
							outputs[3] = highsOptionsToMatlabStruct(highs, false);
						}
					}
				}
				return;
			}
		}
		catch (const matlab::engine::Exception& excpt) {
			error__(excpt.what());
		}
		catch (const matlab::Exception& excpt) {
			error__(excpt.what());
		}
		catch (const std::exception& excpt) {
			error__(excpt.what());
		}
		catch (...) {
			error__("Unexpected error.");
		}
	}

};

// EOF
