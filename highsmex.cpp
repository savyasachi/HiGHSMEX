/* MATLAB mex wrapper over the HiGHS optimization library (https://github.com/ERGO-Code/HiGHS)
 *
 * Author: Savyasachi Singh
 *
 * Covered by the MIT License (see LICENSE file for details).
 * See https://github.com/savyasachi/HiGHSMEX for more information.
 */


// Include the code shared with highsmex_iis. It also pulls in the standard headers,
// the C++ MEX API and HiGHS.
#include "highsmex_common.hpp"
// The mex entry point. It must appear in exactly one translation unit per mex file.
#include "mexAdapter.hpp"


/* ------------------------------------------------------------------------------------------------------ */
/*                                        ENUMERATIONS                                                    */
/* ------------------------------------------------------------------------------------------------------ */

enum class MexCallSyntax { kVer, kDefaultOpts, kIntType, kSolve };


/* ------------------------------------------------------------------------------------------------------ */
/*                                         VARIABLES                                                      */
/* ------------------------------------------------------------------------------------------------------ */

// Const variables
const std::vector<std::string>                linearObjectiveFields({ "weight", "offset", "coefficients", "abs_tolerance", "rel_tolerance", "priority" });
const std::vector<std::string>                highsSolutionFields({ "value_valid", "dual_valid", "col_value", "col_dual", "row_value", "row_dual" });
const std::map<std::string, HighsVarType>     integralityStringsMap({ {"c", HighsVarType::kContinuous}, {"i", HighsVarType::kInteger}, {"sc", HighsVarType::kSemiContinuous}, {"si", HighsVarType::kSemiInteger}, {"ii", HighsVarType::kImplicitInteger} });
const std::map<std::string, HighsBasisStatus> stringToHighsBasisStatusMap({ {"l", HighsBasisStatus::kLower}, {"b", HighsBasisStatus::kBasic}, {"u", HighsBasisStatus::kUpper}, {"z", HighsBasisStatus::kZero}, {"n", HighsBasisStatus::kNonbasic} });
const std::map<HighsBasisStatus, std::string> highsBasisStatusToStringMap({ {HighsBasisStatus::kLower, "l"}, {HighsBasisStatus::kBasic, "b"}, {HighsBasisStatus::kUpper, "u"}, {HighsBasisStatus::kZero, "z"}, {HighsBasisStatus::kNonbasic, "n"} });
const std::vector<std::string>                matlabBasisStructFields({ "valid", "col_status", "row_status" });



TypedArray<MATLABString> highsBasisStatusVectorToMatlabVector(const std::vector<HighsBasisStatus>& v, const bool rowShape) {
	auto out = factory.createArray<MATLABString>(rowShape ? ArrayDimensions({ 1, v.size() }) : ArrayDimensions({ v.size(), 1 }));
	std::transform(v.begin(), v.end(), out.begin(),
		[](const HighsBasisStatus& s_) {
			auto const it_ = highsBasisStatusToStringMap.find(s_); // Here, we are sure that s_ exists in the map
			return it_->second;
		});
	return out;
}


std::vector<HighsBasisStatus> matlabBasisStatusVectorToStdVector(const TypedArray<MATLABString>& arr,
	const std::string& basisStructFieldname, const std::string& mexArgInNumberAsStr) {
	std::vector<HighsBasisStatus> out(numel(arr));
	for (size_t i = 0; i < out.size(); ++i) {
		auto const basisStatusStr = matlabStringToStdString(arr[i]);
		auto const it_ = stringToHighsBasisStatusMap.find(basisStatusStr);
		if (it_ == stringToHighsBasisStatusMap.end()) {
			throw std::runtime_error(formatMessage("Field \"", basisStructFieldname, "\" of the basis struct passed as the ", mexArgInNumberAsStr, " input argument has invalid status string at index ", i + 1, ". \"", basisStatusStr, "\" is not a valid basis status string.")); // Add 1 to match MATLAB indexing
		}
		out[i] = it_->second;
	}
	return out;
}


// Convert HighsBasis to MATLAB struct.
StructArray highsBasisToMatlabStruct(const Highs& highs) {
	auto out = factory.createStructArray({ 1, 1 }, { "valid", "col_status", "row_status" });
	auto const& basis = highs.getBasis();
	out[0]["valid"] = factory.createScalar(basis.valid);
	out[0]["col_status"] = highsBasisStatusVectorToMatlabVector(basis.col_status, false);
	out[0]["row_status"] = highsBasisStatusVectorToMatlabVector(basis.row_status, false);

	return out;
}


// Convert MATLAB struct to HighsBasis.
// Pre-condition: matStruct is a 1x1 struct 
HighsBasis matlabStructToHighsBasis(const StructArray& matStruct, const std::string& mexArgInNumberAsStr) {
	if (!isEqualFieldnames(matlabBasisStructFields, getFieldNames(matStruct))) {
		throw std::runtime_error(formatMessage("Invalid basis struct passed as ", mexArgInNumberAsStr, " input argument."));
	}

	HighsBasis out;
	{
		throwIfInvalidFieldValue(matStruct, 0, "valid", ArrayType::LOGICAL, isScalar,
			formatMessage("Field \"valid\" of the basis struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of logical type."));
		const TypedArray<bool> arr = matStruct[0]["valid"];
		out.valid = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "col_status", ArrayType::MATLAB_STRING, isVectorArr,
			formatMessage("Field \"col_status\" of the basis struct passed as the ", mexArgInNumberAsStr, " input argument must be a vector of MATLAB strings."));
		const TypedArray<MATLABString> arr = matStruct[0]["col_status"];
		out.col_status = matlabBasisStatusVectorToStdVector(arr, "col_status", mexArgInNumberAsStr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "row_status", ArrayType::MATLAB_STRING, isVectorArr,
			formatMessage("Field \"row_status\" of the basis struct passed as the ", mexArgInNumberAsStr, " input argument must be a vector of MATLAB strings."));
		const TypedArray<MATLABString> arr = matStruct[0]["row_status"];
		out.row_status = matlabBasisStatusVectorToStdVector(arr, "row_status", mexArgInNumberAsStr);
	}
	return out;
}


// Pre-condition: indx < numel(matStruct)
void matlabStructToHighsLinearObjective(HighsLinearObjective& out, const StructArray& matStruct, const size_t indx, const std::string& mexArgInNumberAsStr) {
	if (!isEqualFieldnames(linearObjectiveFields, getFieldNames(matStruct))) {
		throw std::runtime_error(formatMessage("Invalid linear objective struct array passed as ", mexArgInNumberAsStr, " input argument."));
	}

	{
		throwIfInvalidFieldValue(matStruct, indx, "weight", ArrayType::DOUBLE, isScalar,
			formatMessage("Field \"weight\" of the linear objective struct at index ", indx + 1, " of the ", mexArgInNumberAsStr, " input argument must be a scalar of double type.")); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["weight"];
		out.weight = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "offset", ArrayType::DOUBLE, isScalar,
			formatMessage("Field \"offset\" of the linear objective struct at index ", indx + 1, " of the ", mexArgInNumberAsStr, " input argument must be a scalar of double type.")); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["offset"];
		out.offset = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "coefficients", ArrayType::DOUBLE, isVectorArr,
			formatMessage("Field \"coefficients\" of the linear objective struct at index ", indx + 1, " of the ", mexArgInNumberAsStr, " input argument must be a vector of double type.")); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["coefficients"];
		out.coefficients = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "abs_tolerance", ArrayType::DOUBLE, isScalar,
			formatMessage("Field \"abs_tolerance\" of the linear objective struct at index ", indx + 1, " of the ", mexArgInNumberAsStr, " input argument must be a scalar of double type.")); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["abs_tolerance"];
		out.abs_tolerance = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "rel_tolerance", ArrayType::DOUBLE, isScalar,
			formatMessage("Field \"rel_tolerance\" of the linear objective struct at index ", indx + 1, " of the ", mexArgInNumberAsStr, " input argument must be a scalar of double type.")); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["rel_tolerance"];
		out.rel_tolerance = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "priority", HighsInt2MatlabArrayType, isScalar,
			formatMessage("Field \"priority\" of the linear objective struct at index ", indx + 1, " of the ", mexArgInNumberAsStr, " input argument must be a scalar of ", HighsInt2MatlabClassStr, " type.")); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<HighsInt> arr = matStruct[indx]["priority"];
		out.priority = arr[0];
	}
}


// Convert HighsSolution to MATLAB struct.
StructArray highsSolutionToMatlabStruct(const Highs& highs) {
	auto out = factory.createStructArray({ 1, 1 }, highsSolutionFields);
	auto const& soln = highs.getSolution();
	out[0]["value_valid"] = factory.createScalar(soln.value_valid);
	out[0]["dual_valid"] = factory.createScalar(soln.dual_valid);
	out[0]["col_value"] = stdVectorToMatlabVector(soln.col_value, false);
	out[0]["col_dual"] = stdVectorToMatlabVector(soln.col_dual, false);
	out[0]["row_value"] = stdVectorToMatlabVector(soln.row_value, false);
	out[0]["row_dual"] = stdVectorToMatlabVector(soln.row_dual, false);
	return out;
}


// Pre-condition: matStruct is a 1x1 struct
HighsSolution matlabStructToHighsSolution(const StructArray& matStruct, const std::string& mexArgInNumberAsStr) {
	if (!isEqualFieldnames(highsSolutionFields, getFieldNames(matStruct))) {
		throw std::runtime_error(formatMessage("Invalid solution struct passed as ", mexArgInNumberAsStr, " input argument."));
	}

	HighsSolution out;
	{
		throwIfInvalidFieldValue(matStruct, 0, "value_valid", ArrayType::LOGICAL, isScalar,
			formatMessage("Field \"value_valid\" of the solution struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of logical type."));
		const TypedArray<bool> arr = matStruct[0]["value_valid"];
		out.value_valid = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "dual_valid", ArrayType::LOGICAL, isScalar,
			formatMessage("Field \"dual_valid\" of the solution struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of logical type."));
		const TypedArray<bool> arr = matStruct[0]["dual_valid"];
		out.dual_valid = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "col_value", ArrayType::DOUBLE, isVectorArr,
			formatMessage("Field \"col_value\" of the solution struct passed as the ", mexArgInNumberAsStr, " input argument must be a vector of double type."));
		const TypedArray<double> arr = matStruct[0]["col_value"];
		out.col_value = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "col_dual", ArrayType::DOUBLE, isVectorArr,
			formatMessage("Field \"col_dual\" of the solution struct passed as the ", mexArgInNumberAsStr, " input argument must be a vector of double type."));
		const TypedArray<double> arr = matStruct[0]["col_dual"];
		out.col_dual = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "row_value", ArrayType::DOUBLE, isVectorArr,
			formatMessage("Field \"row_value\" of the solution struct passed as the ", mexArgInNumberAsStr, " input argument must be a vector of double type."));
		const TypedArray<double> arr = matStruct[0]["row_value"];
		out.row_value = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "row_dual", ArrayType::DOUBLE, isVectorArr,
			formatMessage("Field \"row_dual\" of the solution struct passed as the ", mexArgInNumberAsStr, " input argument must be a vector of double type."));
		const TypedArray<double> arr = matStruct[0]["row_dual"];
		out.row_dual = matlabVectorToStdVector(arr);
	}
	return out;
}


/* ------------------------------------------------------------------------------------------------------ */
/*                                         MEX INTERFACE                                                  */
/* ------------------------------------------------------------------------------------------------------ */

struct process1stArgInResults {
	bool isMultiObjective = false; // true if the first input argument is a struct array of linear objectives
	std::vector<double> colCost; // Column costs if the first input argument is a vector of doubles, or a cell array of 2 elements
	double offset = 0; // Offset if the first input argument is a cell array of 2 elements
	std::vector<HighsLinearObjective> linearObjectives; // Linear objectives if the first input argument is a struct array of linear objectives
};


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
		else if (inputs.size() >= 4 && inputs.size() <= 12) {
			return MexCallSyntax::kSolve;
		}
		else {
			throw std::runtime_error("Invalid number of input arguments.");
		}
	}

	// Pre-condition: 1) inputs.size()>0
	process1stArgInResults process1stArgIn(ArgumentList& inputs) {
		process1stArgInResults out;
		auto const dims = inputs[0].getDimensions();
		auto const numelIn0 = numel(inputs[0]);
		switch (getType(inputs[0])) {
		case ArrayType::DOUBLE:
		{
			if (!isVector(dims)) {
				throw std::runtime_error("First input argument (c) must be a double type vector.");
			}
			const TypedArray<double> c(inputs[0]);
			out.isMultiObjective = false;
			out.colCost = matlabVectorToStdVector(c);
			out.offset = 0;
			break;
		}

		case ArrayType::CELL:
		{
			if (numelIn0 != 2) {
				throw std::runtime_error("First input argument (c) must be a 1x2 or 2x1 cell array.");
			}
			const CellArray cell(inputs[0]);
			auto const dimsCell0 = cell[0].getDimensions();
			if (!(isDouble(cell[0]) && isVector(dimsCell0))) {
				throw std::runtime_error("The first element of the cell array passed as the first input argument (c) must be a double type vector.");
			}
			const TypedArray<double> c = cell[0];
			if (!(isDouble(cell[1]) && isScalar(cell[1]))) {
				throw std::runtime_error("The second element of the cell array passed as the first input argument (c) must be a double scalar.");
			}
			const TypedArray<double> offset = cell[1];
			out.isMultiObjective = false;
			out.colCost = matlabVectorToStdVector(c);
			out.offset = offset[0];
			break;
		}

		case ArrayType::STRUCT:
		{
			if (!isVector(dims)) {
				throw std::runtime_error("First input argument (c) must be a MATLAB struct array representing the multiple linear objectives.");
			}
			const StructArray linObjStructs(inputs[0]);
			std::vector<HighsLinearObjective> linearObjectives(numelIn0);
			for (size_t i = 0; i < linearObjectives.size(); ++i) {
				matlabStructToHighsLinearObjective(linearObjectives[i], linObjStructs, i, "first");
			}
			out.isMultiObjective = true;
			out.linearObjectives = std::move(linearObjectives);
			break;
		}

		default:
			throw std::runtime_error("First input argument (c) must be a double type vector, or a cell array, or a MATLAB struct array.");
		}

		return out;
	}

	// Pre-condition: 1) inputs.size()>1, 2) highsModel.lp_.num_col_ must be set.
	void process2ndArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		if (isEmpty(inputs[1])) { // No linear constraints, hence no matrix A
			highsModel.lp_.num_row_ = 0;
			highsModel.lp_.a_matrix_.start_.assign(highsModel.lp_.num_col_ + 1, 0);
			return;
		}

		bool isSparse = false;
		if (isDouble(inputs[1])) {
			auto const dims = inputs[1].getDimensions();
			if (!(isMatrix(dims) && highsModel.lp_.num_col_ == castToHighsInt(dims[1]))) {
				throw std::runtime_error(formatMessage("Second input argument (A) must be a matrix of double type with ", highsModel.lp_.num_col_, " columns."));
			}
			highsModel.lp_.num_row_ = castToHighsInt(dims[0]);
		}
		else if (isCell(inputs[1])) {
			if (numel(inputs[1]) != 5) throw std::runtime_error("Second input argument (A) must be a cell array of 5 elements.");
			isSparse = true;
			const CellArray cell(inputs[1]);
			// Retrieve the matrix dimensions from the cell array						
			if (!(isDouble(cell[3]) && isScalar(cell[3]) && isDouble(cell[4]) && isScalar(cell[4]))) {
				throw std::runtime_error("The 4th and 5th elements of the cell array passed as the second input argument (A) must be double scalars representing the number of rows and columns of A respectively.");
			}
			const TypedArray<double> nrowsA = cell[3];
			const TypedArray<double> ncolsA = cell[4];
			if (highsModel.lp_.num_col_ != castToHighsInt(ncolsA[0])) {
				throw std::runtime_error(formatMessage("The 5th element of the cell array passed as the second input argument (A) must be a double scalar equal to ", highsModel.lp_.num_col_, "."));
			}
			highsModel.lp_.num_row_ = castToHighsInt(nrowsA[0]);
		}
		else {
			throw std::runtime_error("Second input argument (A) must be a matrix of double type or, a cell array of 5 elements.");
		}

		highsModel.lp_.a_matrix_.format_ = MatrixFormat::kColwise;
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
			matlabMatrixToHighsFormat(
				highsModel.lp_.a_matrix_.start_,
				highsModel.lp_.a_matrix_.index_,
				highsModel.lp_.a_matrix_.value_,
				iA,
				jA,
				vA,
				highsModel.lp_.num_row_,
				highsModel.lp_.num_col_,
				false);
		}
		else {
			const TypedArray<double> A(inputs[1]);
			matlabMatrixToHighsFormat(
				highsModel.lp_.a_matrix_.start_,
				highsModel.lp_.a_matrix_.index_,
				highsModel.lp_.a_matrix_.value_,
				A,
				highsModel.lp_.num_row_,
				highsModel.lp_.num_col_,
				false);
		}

		if constexpr (MexDebugPrinting) {
			print("lp_.a_matrix_.start_ = "); disp(highsModel.lp_.a_matrix_.start_);
			print("lp_.a_matrix_.index_ = "); disp(highsModel.lp_.a_matrix_.index_);
			print("lp_.a_matrix_.value_ = "); disp(highsModel.lp_.a_matrix_.value_);
		}
	}


	// Pre-condition: 1) inputs.size()>2, 2) highsModel.lp_.num_row_ must be set.
	void process3rdArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		if (!highsModel.lp_.num_row_) { // No linear constraints, hence no L
			if (isEmpty(inputs[2])) {
				return; // No lower bounds on the rows, hence no need to set them
			}
			else {
				throw std::runtime_error("Third input argument (L) must be an empty array when there are no linear constraints.");
			}
		}

		bool setToDefault = true;
		if (!isEmpty(inputs[2])) {
			auto const dims = inputs[2].getDimensions();
			if (!(isDouble(inputs[2]) && isVector(dims))) {
				throw std::runtime_error("Third input argument (L) must be a vector of double type, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			highsModel.lp_.row_lower_.assign(highsModel.lp_.num_row_, -kHighsInf);
		}
		else {
			const TypedArray<double> L(inputs[2]);
			if (numel(L) != highsModel.lp_.num_row_) {
				throw std::runtime_error(formatMessage("Expected length of third input argument (L) to be ", highsModel.lp_.num_row_, "."));
			}
			highsModel.lp_.row_lower_ = matlabVectorToStdVector(L);
		}

		if constexpr (MexDebugPrinting) {
			print("lp_.row_lower_ = "); disp(highsModel.lp_.row_lower_);
		}
	}

	// Pre-condition: 1) inputs.size()>3, 2) highsModel.lp_.num_row_ must be set.
	void process4thArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		if (!highsModel.lp_.num_row_) { // No linear constraints, hence no U
			if (isEmpty(inputs[3])) {
				return; // No upper bounds on the rows, hence no need to set them
			}
			else {
				throw std::runtime_error("Fourth input argument (U) must be an empty array when there are no linear constraints.");
			}
		}

		bool setToDefault = true;
		if (!isEmpty(inputs[3])) {
			auto const dims = inputs[3].getDimensions();
			if (!(isDouble(inputs[3]) && isVector(dims))) {
				throw std::runtime_error("Fourth input argument (U) must be a vector of double type, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			highsModel.lp_.row_upper_.assign(highsModel.lp_.num_row_, kHighsInf);
		}
		else {
			const TypedArray<double> U(inputs[3]);
			if (numel(U) != highsModel.lp_.num_row_) {
				throw std::runtime_error(formatMessage("Expected length of fourth input argument (U) to be ", highsModel.lp_.num_row_, "."));
			}
			highsModel.lp_.row_upper_ = matlabVectorToStdVector(U);
		}

		if constexpr (MexDebugPrinting) {
			print("lp_.row_upper_ = "); disp(highsModel.lp_.row_upper_);
		}
	}

	// Pre-condition: highsModel.lp_.num_col_ must be set.
	void process5thArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		bool setToDefault = true;
		if (inputs.size() > 4 && !isEmpty(inputs[4])) {
			auto const dims = inputs[4].getDimensions();
			if (!(isDouble(inputs[4]) && isVector(dims))) {
				throw std::runtime_error("Fifth input argument (l) must be a vector of double type, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			highsModel.lp_.col_lower_.assign(highsModel.lp_.num_col_, -kHighsInf);
		}
		else {
			const TypedArray<double> l(inputs[4]);
			if (numel(l) != highsModel.lp_.num_col_) {
				throw std::runtime_error(formatMessage("Expected length of fifth input argument (l) to be ", highsModel.lp_.num_col_, "."));
			}
			highsModel.lp_.col_lower_ = matlabVectorToStdVector(l);
		}

		if constexpr (MexDebugPrinting) {
			print("lp_.col_lower_ = "); disp(highsModel.lp_.col_lower_);
		}
	}

	// Pre-condition: highsModel.lp_.num_col_ must be set.
	void process6thArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		bool setToDefault = true;
		if (inputs.size() > 5 && !isEmpty(inputs[5])) {
			auto const dims = inputs[5].getDimensions();
			if (!(isDouble(inputs[5]) && isVector(dims))) {
				throw std::runtime_error("Sixth input argument (u) must be a vector of double type, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			highsModel.lp_.col_upper_.assign(highsModel.lp_.num_col_, kHighsInf);
		}
		else {
			const TypedArray<double> u(inputs[5]);
			if (numel(u) != highsModel.lp_.num_col_) {
				throw std::runtime_error(formatMessage("Expected length of sixth input argument (u) to be ", highsModel.lp_.num_col_, "."));
			}
			highsModel.lp_.col_upper_ = matlabVectorToStdVector(u);
		}

		if constexpr (MexDebugPrinting) {
			print("lp_.col_upper_ = "); disp(highsModel.lp_.col_upper_);
		}
	}

	void process7thArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		bool setToDefault = true;
		bool isSparse = false;
		if (inputs.size() > 6 && !isEmpty(inputs[6])) {
			setToDefault = false;
			if (isDouble(inputs[6])) {
				auto const dims = inputs[6].getDimensions();
				if (!isSquareMatrix(dims)) throw std::runtime_error("Seventh input argument (Q) must be a square matrix of double type.");
				highsModel.hessian_.dim_ = castToHighsInt(dims[0]);
			}
			else if (isCell(inputs[6])) {
				if (numel(inputs[6]) != 5) throw std::runtime_error("Seventh input argument (Q) must be a cell array of 5 elements.");
				isSparse = true;
				const CellArray cell(inputs[6]);
				// Retrieve the matrix dimensions from the cell array						
				if (!(isDouble(cell[3]) && isScalar(cell[3]) && isDouble(cell[4]) && isScalar(cell[4]))) {
					throw std::runtime_error("The 4th and 5th elements of the cell array passed as the Seventh input argument (Q) must be double scalars representing the number of rows and columns of Q respectively.");
				}
				const TypedArray<double> nrowsQ = cell[3];
				const TypedArray<double> ncolsQ = cell[4];
				if (!(castToHighsInt(ncolsQ[0]) == castToHighsInt(nrowsQ[0]))) {
					throw std::runtime_error("Seventh input argument (Q) must be a square matrix.");
				}
				highsModel.hessian_.dim_ = castToHighsInt(ncolsQ[0]);
			}
			else {
				throw std::runtime_error("Seventh input argument (Q) must be a matrix of double type or, a cell array of 5 elements.");
			}
		}

		highsModel.hessian_.format_ = HessianFormat::kTriangular;
		if (setToDefault) {
			highsModel.hessian_.dim_ = 0;
		}
		else {
			if (highsModel.hessian_.dim_ != highsModel.lp_.num_col_) {
				throw std::runtime_error(formatMessage("Expected dimension of the seventh input argument (Q) to be ", highsModel.lp_.num_col_, "."));
			}
			if (isSparse) {
				const CellArray cell(inputs[6]);
				auto isDoubleVec = [](const Array& arr_) -> bool { return isDouble(arr_) && (isEmpty(arr_) || isVectorArr(arr_)); };
				if (!(isDoubleVec(cell[0]) && isDoubleVec(cell[1]) && isDoubleVec(cell[2]))) {
					throw std::runtime_error("The 1st, 2nd, and 3rd elements of the cell array passed as the seventh input argument (Q) must be double type vectors.");
				}
				const TypedArray<double> iQ = cell[0];
				const TypedArray<double> jQ = cell[1];
				const TypedArray<double> vQ = cell[2];
				auto const nQ = numel(iQ);
				if (!(nQ == numel(jQ) && nQ == numel(vQ))) {
					throw std::runtime_error("The 1st, 2nd, and 3rd elements of the cell array passed as the seventh input argument (Q) must be vectors of the same length.");
				}
				auto const nnz = matlabMatrixToHighsFormat(
					highsModel.hessian_.start_,
					highsModel.hessian_.index_,
					highsModel.hessian_.value_,
					iQ,
					jQ,
					vQ,
					highsModel.hessian_.dim_,
					highsModel.hessian_.dim_,
					true);
				if (!nnz) highsModel.hessian_.dim_ = 0; // If the Hessian is all zeros then set the dimension to 0
			}
			else {
				const TypedArray<double> Q(inputs[6]);
				auto const nnz = matlabMatrixToHighsFormat(
					highsModel.hessian_.start_,
					highsModel.hessian_.index_,
					highsModel.hessian_.value_,
					Q,
					highsModel.hessian_.dim_,
					highsModel.hessian_.dim_,
					true);
				if (!nnz) highsModel.hessian_.dim_ = 0; // If the Hessian is all zeros then set the dimension to 0
			}
		}

		if constexpr (MexDebugPrinting) {
			print("hessian_.start_ = "); disp(highsModel.hessian_.start_);
			print("hessian_.index_ = "); disp(highsModel.hessian_.index_);
			print("hessian_.value_ = "); disp(highsModel.hessian_.value_);
		}
	}


	// Pre-condition: highsModel.lp_.num_col_ must be set.
	void process8thArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		bool setToDefault = true;
		if (inputs.size() > 7 && !isEmpty(inputs[7])) {
			auto const dims = inputs[7].getDimensions();
			if (!(isMatlabString(inputs[7]) && isVector(dims))) {
				throw std::runtime_error("Eighth input argument (integrality) must be a vector of MATLAB strings, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			// Do nothing. By default integrality is not set.
			// NOTE: If we explicitly set all the variables as continuous here then HiGHS emits the following warning.
			//       "WARNING: No semi-integer/integer variables in model with non-empty integrality"
		}
		else {
			const TypedArray<MATLABString> integralityStrings(inputs[7]);
			if (numel(integralityStrings) != highsModel.lp_.num_col_) {
				throw std::runtime_error(formatMessage("Expected length of the eighth input argument (integrality) to be ", highsModel.lp_.num_col_, "."));
			}
			highsModel.lp_.integrality_.resize(numel(integralityStrings));
			for (size_t i = 0; i < numel(integralityStrings); ++i) {
				auto const integralityStr = matlabStringToStdString(integralityStrings[i]);
				auto const it = integralityStringsMap.find(integralityStr);
				if (it == integralityStringsMap.end()) {
					throw std::runtime_error(formatMessage("Invalid string at index ", i + 1, " of the eighth input argument (integrality). \"", integralityStr, "\" is not a valid integrality string.")); // Add 1 to the index to match MATLAB's indexing
				}
				highsModel.lp_.integrality_[i] = it->second;
			}
		}

		if constexpr (MexDebugPrinting) {
			std::vector<HighsInt> tmp(highsModel.lp_.integrality_.size());
			for (size_t i = 0; i < tmp.size(); ++i) {
				tmp[i] = castToHighsInt(highsModel.lp_.integrality_[i]);
			}
			print("lp_.integrality_ = "); disp(tmp);
		}
	}

	void process9thArgIn(ArgumentList& inputs, Highs& highs) {
		bool setToDefault = true;
		if (inputs.size() > 8 && !isEmpty(inputs[8])) {
			if (!(isStruct(inputs[8]) && isScalar(inputs[8]))) {
				throw std::runtime_error("Ninth input argument (options) must be a 1x1 MATLAB struct, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			// Do nothing because a newly constructed Highs instance is populated with default HiGHS option values
		}
		else {
			const StructArray matOptsStruct(inputs[8]);
			setHighsOptions(highs, matOptsStruct, "ninth");
		}
	}

	void process10thArgIn(ArgumentList& inputs, HighsModel& highsModel) {
		bool setToDefault = true;
		if (inputs.size() > 9 && !isEmpty(inputs[9])) {
			if (!(isMatlabString(inputs[9]) && isScalar(inputs[9]))) {
				throw std::runtime_error("Tenth input argument (objSense) must be a MATLAB string, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			highsModel.lp_.sense_ = ObjSense::kMinimize;
		}
		else {
			const TypedArray<MATLABString> in9(inputs[9]);
			auto const objSenseStr = matlabStringToStdString(in9[0]);
			if (objSenseStr == "min") {
				highsModel.lp_.sense_ = ObjSense::kMinimize;
			}
			else if (objSenseStr == "max") {
				highsModel.lp_.sense_ = ObjSense::kMaximize;
			}
			else {
				throw std::runtime_error("Invalid MATLAB string passed as tenth input argument (objSense). It should be \"max\" or \"min\".");
			}
		}

		if constexpr (MexDebugPrinting) {
			print(formatMessage("lp_.sense_ = ", highsModel.lp_.sense_ == ObjSense::kMinimize ? "kMinimize" : "kMaximize", "\n"));
		}
	}

	void process11thArgIn(ArgumentList& inputs, Highs& highs, const HighsInt numCol) {
		bool setToDefault = true;
		bool isMatStruct = false;
		if (inputs.size() > 10 && !isEmpty(inputs[10])) {
			auto const dims = inputs[10].getDimensions();
			isMatStruct = isStruct(inputs[10]);
			if (!(
				(isMatStruct && isScalar(inputs[10])) || (isDouble(inputs[10]) && isVector(dims))
				)) {
				throw std::runtime_error("Eleventh input argument (setSoln) must be a 1x1 MATLAB struct, or a double type vector, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			// Do nothing. By default no hot starting is performed.
		}
		else {
			// Call Highs::setSolution 
			if (isMatStruct) {
				const StructArray matSoln0Struct(inputs[10]);
				auto const soln0 = matlabStructToHighsSolution(matSoln0Struct, "eleventh");
				checkHighsReturnStatus(highs.setSolution(soln0),
					"Warning issued when setting the solution struct in the eleventh input argument (setSoln) with the HiGHS solver.",
					"Failed to set solution struct in the eleventh input argument (setSoln) with the HiGHS solver.");
			}
			else {
				// Received the primal solution as a double vector. Convert it to the sparse representation.				
				const TypedArray<double> soln0(inputs[10]);
				auto const n = numel(soln0);
				if (n != numCol) {
					throw std::runtime_error(formatMessage("Expected length of the eleventh input argument (setSoln) to be ", numCol, "."));
				}
				std::vector<HighsInt> index;
				std::vector<double> value;
				index.reserve(n);
				value.reserve(n);
				HighsInt numEntries = 0;
				for (size_t i = 0; i < n; ++i) {
					if (!soln0[i]) continue;
					++numEntries;
					index.push_back(castToHighsInt(i));
					value.push_back(soln0[i]);
				}
				checkHighsReturnStatus(highs.setSolution(numEntries, index.data(), value.data()),
					"Warning issued when setting the solution vector in the eleventh input argument (setSoln) with the HiGHS solver.",
					"Failed to set solution vector in the eleventh input argument (setSoln) with the HiGHS solver.");
			}
		}

		if constexpr (MexDebugPrinting) {
			auto const tmp = highsSolutionToMatlabStruct(highs);
			print("highs solution after setSolution = \n"); disp(tmp);
		}
	}

	void process12thArgIn(ArgumentList& inputs, Highs& highs) {
		bool setToDefault = true;
		if (inputs.size() > 11 && !isEmpty(inputs[11])) {
			if (!(isStruct(inputs[11]) && isScalar(inputs[11]))) {
				throw std::runtime_error("Twelfth input argument (setBasis) must be a 1x1 MATLAB struct, or an empty array.");
			}
			setToDefault = false;
		}

		if (setToDefault) {
			// Do nothing. By default no basis is set.
		}
		else {
			// Call Highs::setBasis 
			const StructArray matBasisStruct(inputs[11]);
			auto const basis = matlabStructToHighsBasis(matBasisStruct, "twelfth");
			checkHighsReturnStatus(highs.setBasis(basis),
				"Warning issued when setting the basis in the twelfth input argument (setBasis) with the HiGHS solver.",
				"Failed to set basis in the twelfth input argument (setBasis) with the HiGHS solver.");
		}

		if constexpr (MexDebugPrinting) {
			auto const tmp = highsBasisToMatlabStruct(highs);
			print("highs basis after setBasis = \n"); disp(tmp);
		}
	}

	void runSolver(ArgumentList& inputs, Highs& highs, HighsModel& highsModel) {
		// Process input arguments
		auto proc1stArgResults = process1stArgIn(inputs);
		if (proc1stArgResults.isMultiObjective) {
			highsModel.lp_.num_col_ = castToHighsInt(proc1stArgResults.linearObjectives[0].coefficients.size());
			highsModel.lp_.col_cost_.assign(highsModel.lp_.num_col_, 0);
			highsModel.lp_.offset_ = 0;
		}
		else {
			highsModel.lp_.num_col_ = castToHighsInt(proc1stArgResults.colCost.size());
			highsModel.lp_.col_cost_ = std::move(proc1stArgResults.colCost);
			highsModel.lp_.offset_ = proc1stArgResults.offset;
		}
		process2ndArgIn(inputs, highsModel);
		process3rdArgIn(inputs, highsModel);
		process4thArgIn(inputs, highsModel);
		process5thArgIn(inputs, highsModel);
		process6thArgIn(inputs, highsModel);
		process7thArgIn(inputs, highsModel);
		process8thArgIn(inputs, highsModel);
		process9thArgIn(inputs, highs);
		process10thArgIn(inputs, highsModel);

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

		// Pass constraints and hessian to HiGHS
		checkHighsReturnStatus(highs.passModel(highsModel),
			"Warning issued when passing the model to the HiGHS solver.",
			"Failed to pass the model to the HiGHS solver.");

		// Set multiple objectives
		if (proc1stArgResults.isMultiObjective) {
			checkHighsReturnStatus(highs.passLinearObjectives(
				castToHighsInt(proc1stArgResults.linearObjectives.size()),
				proc1stArgResults.linearObjectives.data()),
				"Warning issued when passing multiple linear objectives in the first input argument (c) to the HiGHS solver.",
				"Failed to pass multiple linear objectives in the first input argument (c) to the HiGHS solver.");
		}

		// Set solution for hot starting
		process11thArgIn(inputs, highs, highsModel.lp_.num_col_); // This must be done after passing the model

		// Set basis
		process12thArgIn(inputs, highs);

		// Run solver
		checkHighsReturnStatus(highs.run(),
			"Warning issued during running the HiGHS solver.",
			"Failure during running the HiGHS solver.");

		// highs.stopCallback(HighsCallbackType::kCallbackLogging); // Not needed. We are exiting after setting outputs
	}

public:

	/* This is the gateway routine for the MEX-file. */
	void operator()(ArgumentList outputs, ArgumentList inputs) {
		try {
			HighsModel highsModel;
			Highs highs;

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

			case MexCallSyntax::kSolve:
				if (!(outputs.size() >= 1 && outputs.size() <= 4)) {
					throw std::runtime_error("Number of output arguments must be >= 1 and <= 4.");
				}
				// Clear the error stack
				while (!highsLogErrStack.empty()) {
					highsLogErrStack.pop();
				}
				// Process inputs and run the HiGHS solver
				runSolver(inputs, highs, highsModel);
				// Assign outputs
				if (outputs.size() > 0) {
					outputs[0] = highsSolutionToMatlabStruct(highs);
					if (outputs.size() > 1) {
						outputs[1] = highsInfoToMatlabStruct(highs);
						if (outputs.size() > 2) {
							outputs[2] = highsOptionsToMatlabStruct(highs, false);
							if (outputs.size() > 3) {
								outputs[3] = highsBasisToMatlabStruct(highs);
							}
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
