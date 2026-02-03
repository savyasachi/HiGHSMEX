/* MATLAB mex wrapper over the HiGHS optimization library (https://github.com/ERGO-Code/HiGHS)
 *
 * Author: Savyasachi Singh
 *
 * Covered by the MIT License (see LICENSE file for details).
 * See https://github.com/savyasachi/HiGHSMEX for more information.
 */


 // Include standard headers
#include <string>
#include <vector>	
#include <map>
#include <set>
#include <stack>
#include <exception>
#include <stdexcept>
#include <format>
#include <type_traits>
#include <algorithm>
#include <complex>
// Include C++ MEX API
#include "mex.hpp"
#include "mexAdapter.hpp"
#include "MatlabDataArray.hpp"
// Include HiGHS
#include "Highs.h"


// Open MATLAB namespaces
using namespace matlab::mex;
using namespace matlab::data;
using namespace matlab::engine;


/* ------------------------------------------------------------------------------------------------------ */
/*                                        ENUMERATIONS                                                    */
/* ------------------------------------------------------------------------------------------------------ */

enum class MexCallSyntax { kVer, kDefaultOpts, kIntType, kSolve };


/* ------------------------------------------------------------------------------------------------------ */
/*                                         VARIABLES                                                      */
/* ------------------------------------------------------------------------------------------------------ */

// Compile time constants
constexpr ArrayType   HighsInt2MatlabArrayType = std::is_same_v<HighsInt, int32_t> ? ArrayType::INT32 : ArrayType::INT64;
constexpr std::string HighsInt2MatlabClassStr = std::is_same_v<HighsInt, int32_t> ? "int32" : "int64";
constexpr bool        MexDebugPrinting = 0;

// Const variables
const std::vector<std::string>                linearObjectiveFields({ "weight", "offset", "coefficients", "abs_tolerance", "rel_tolerance", "priority" });
const std::vector<std::string>                highsSolutionFields({ "value_valid", "dual_valid", "col_value", "col_dual", "row_value", "row_dual" });
const std::map<std::string, HighsVarType>     integralityStringsMap({ {"c", HighsVarType::kContinuous}, {"i", HighsVarType::kInteger}, {"sc", HighsVarType::kSemiContinuous}, {"si", HighsVarType::kSemiInteger}, {"ii", HighsVarType::kImplicitInteger} });
const std::map<std::string, HighsBasisStatus> stringToHighsBasisStatusMap({ {"l", HighsBasisStatus::kLower}, {"b", HighsBasisStatus::kBasic}, {"u", HighsBasisStatus::kUpper}, {"z", HighsBasisStatus::kZero}, {"n", HighsBasisStatus::kNonbasic} });
const std::map<HighsBasisStatus, std::string> highsBasisStatusToStringMap({ {HighsBasisStatus::kLower, "l"}, {HighsBasisStatus::kBasic, "b"}, {HighsBasisStatus::kUpper, "u"}, {HighsBasisStatus::kZero, "z"}, {HighsBasisStatus::kNonbasic, "n"} });
const std::vector<std::string>                matlabBasisStructFields({ "valid", "col_status", "row_status" });

// Variables
ArrayFactory factory;


/* ------------------------------------------------------------------------------------------------------ */
/*                                          FUNCTIONS                                                     */
/* ------------------------------------------------------------------------------------------------------ */

template <typename T>
constexpr HighsInt castToHighsInt(const T x) {
	return static_cast<HighsInt>(x);
}


// Returns the type of the MATLAB array.
inline ArrayType getType(const Array& arr) noexcept(noexcept(arr.getType())) {
	return arr.getType();
}


// Returns true if the input MATLAB array (non-sparse) is double type.
inline bool isDouble(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::DOUBLE;
}


// Returns true if the input MATLAB array is a struct.
inline bool isStruct(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::STRUCT;
}


// Returns true if the input MATLAB array is a cell.
inline bool isCell(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::CELL;
}


// Returns true if the input MATLAB array is a MATLAB string (delimited by double quotes "").
inline bool isMatlabString(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::MATLAB_STRING;
}


// Returns number of elements in a matlab::data::Array.
inline size_t numel(const Array& arr) noexcept(noexcept(arr.getNumberOfElements())) {
	return arr.getNumberOfElements();
}


// Returns true if the input matlab::data::Array is empty.
inline bool isEmpty(const Array& arr) noexcept(noexcept(arr.isEmpty())) {
	return arr.isEmpty();
}


// Returns true if the input matlab::data::Array has size 1.
inline bool isScalar(const Array& arr) noexcept(noexcept(numel(arr))) {
	return numel(arr) == 1;
}


// Returns true if the input ArrayDimensions is a vector.
inline bool isVector(const ArrayDimensions& dims) {
	return dims.size() == 2 && ((dims[0] == 1 && dims[1] > 0) || (dims[0] > 0 && dims[1] == 1));
}


// Returns true if the input matlab::data::Array is a vector.
inline bool isVectorArr(const Array& arr) {
	return isVector(arr.getDimensions());
}


// Returns true if the input ArrayDimensions is a matrix.
inline bool isMatrix(const ArrayDimensions& dims) {
	return dims.size() == 2 && dims[0] > 0 && dims[1] > 0;
}


// Returns true if the input ArrayDimensions is a matrix.
inline bool isSquareMatrix(const ArrayDimensions& dims) {
	return dims.size() == 2 && dims[0] > 0 && dims[0] == dims[1];
}


// Get the names of fields of a struct array.
inline std::vector<std::string> getFieldNames(const StructArray& arr) {
	auto frange = arr.getFieldNames();
	return { frange.begin(), frange.end() };
}


// Compare fieldnames of the first struct with the fieldnames of the second struct for equality.
bool isEqualFieldnames(const std::vector<std::string>& f1, const std::vector<std::string>& f2) {
	if (f1.size() != f2.size()) return false;
	const std::set<std::string> f2set(f2.begin(), f2.end());
	for (auto const& s : f1) {
		if (!f2set.contains(s)) return false;
	}
	return true;
}


// Convert MATLAB string to std::string.
inline std::string matlabStringToStdString(const MATLABString& matlabStr) {
	return matlabStr.has_value() ? convertUTF16StringToUTF8String(*matlabStr) : "";
}


template <typename>
struct is_std_complex : public std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : public std::true_type {};

// Extracts the pointer to underlying data from the non-const iterator (`TypedIterator<T>`).
/* This function does not throw any exceptions. */
template <typename T>
inline T* toPointer(const matlab::data::TypedIterator<T>& it) noexcept(noexcept(it.operator->())) {
	static_assert((std::is_arithmetic<T>::value || is_std_complex<T>::value) && !std::is_const<T>::value,
		"Template argument T must be a std::is_arithmetic type or std::complex.");
	return it.operator->();
}


/* Extracts pointer to the first element in the array.
 * Example usage:
 * ArrayFactory factory;
 * TypedArray<double> A = factory.createArray<double>({ 2,2 }, { 1.0, 3.0, 2.0, 4.0 });
 * auto ptr = getPointer(A);
 * NOTE: Do not call `getPointer` with temporary object. e.g., the following code is ill-formed.
 *       auto ptr=getPointer(factory.createArray<double>({ 2,2 },{ 1.0, 3.0, 2.0, 4.0 }));
 */
template <typename T>
inline T* getPointer(matlab::data::TypedArray<T>& arr) noexcept(noexcept(toPointer(arr.begin()))) {
	return toPointer(arr.begin());
}
template <typename T>
inline const T* getPointer(const matlab::data::TypedArray<T>& arr) noexcept(noexcept(toPointer(arr.begin()))) {
	return getPointer(const_cast<matlab::data::TypedArray<T>&>(arr));
}


// Convert std::vector to MATLAB vector.
template <typename T>
TypedArray<T> stdVectorToMatlabVector(const std::vector<T>& v, const bool rowShape) {
	auto out = factory.createArray<T>(rowShape ? ArrayDimensions({ 1, v.size() }) : ArrayDimensions({ v.size(), 1 }));
	std::copy(v.begin(), v.end(), getPointer(out));
	return out;
}


// Convert MATLAB vector to std::vector.
template <typename T>
inline std::vector<T> matlabVectorToStdVector(const TypedArray<T>& arr) {
	auto pBegin = getPointer(arr);
	return { pBegin, pBegin + numel(arr) };
}


// Pre-condition: indx < numel(matStruct)
inline void throwIfInvalidFieldValue(const StructArray& matStruct, const size_t indx, const std::string& fieldname, const ArrayType fieldType,
	std::function<bool(const Array& arr)> fieldValueCheck, const std::string& errMsg) {
	if (!(getType(matStruct[indx][fieldname]) == fieldType && fieldValueCheck(matStruct[indx][fieldname]))) {
		throw std::runtime_error(errMsg);
	}
}


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
			throw std::runtime_error(std::format("Field \"{}\" of the basis struct passed as the {} input argument has invalid status string at index {}. \"{}\" is not a valid basis status string.",
				basisStructFieldname, mexArgInNumberAsStr, i + 1, basisStatusStr)); // Add 1 to match MATLAB indexing
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
		throw std::runtime_error(std::format("Invalid basis struct passed as {} input argument.", mexArgInNumberAsStr));
	}

	HighsBasis out;
	{
		throwIfInvalidFieldValue(matStruct, 0, "valid", ArrayType::LOGICAL, isScalar,
			std::format("Field \"valid\" of the basis struct passed as the {} input argument must be a scalar of logical type.", mexArgInNumberAsStr));
		const TypedArray<bool> arr = matStruct[0]["valid"];
		out.valid = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "col_status", ArrayType::MATLAB_STRING, isVectorArr,
			std::format("Field \"col_status\" of the basis struct passed as the {} input argument must be a vector of MATLAB strings.", mexArgInNumberAsStr));
		const TypedArray<MATLABString> arr = matStruct[0]["col_status"];
		out.col_status = matlabBasisStatusVectorToStdVector(arr, "col_status", mexArgInNumberAsStr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "row_status", ArrayType::MATLAB_STRING, isVectorArr,
			std::format("Field \"row_status\" of the basis struct passed as the {} input argument must be a vector of MATLAB strings.", mexArgInNumberAsStr));
		const TypedArray<MATLABString> arr = matStruct[0]["row_status"];
		out.row_status = matlabBasisStatusVectorToStdVector(arr, "row_status", mexArgInNumberAsStr);
	}
	return out;
}


// Convert HighsInfo to MATLAB struct. We add some extra fields to the output MATLAB struct.
StructArray highsInfoToMatlabStruct(const Highs& highs) {
	auto out = factory.createStructArray({ 1, 1 },
		{ // These fields mirror the HighsInfo class of the HiGHS library
			"valid", "mip_node_count", "simplex_iteration_count", "ipm_iteration_count", "crossover_iteration_count",
		"pdlp_iteration_count", "qp_iteration_count", "primal_solution_status", "dual_solution_status", "basis_validity",
		"objective_function_value", "mip_dual_bound", "mip_gap", "max_integrality_violation", "num_primal_infeasibilities",
		"max_primal_infeasibility", "sum_primal_infeasibilities", "num_dual_infeasibilities", "max_dual_infeasibility",
		"sum_dual_infeasibilities", "num_relative_primal_infeasibilities", "max_relative_primal_infeasibility",
		"num_relative_dual_infeasibilities", "max_relative_dual_infeasibility", "num_primal_residual_errors",
		"max_primal_residual_error", "num_dual_residual_errors", "max_dual_residual_error",
		"num_relative_primal_residual_errors", "max_relative_primal_residual_error", "num_relative_dual_residual_errors",
		"max_relative_dual_residual_error", "num_complementarity_violations", "max_complementarity_violation",
		"primal_dual_objective_error", "primal_dual_integral",
		// These fields are extra. They are added by highsmex.
		"primal_solution_status_string", "dual_solution_status_string", "basis_validity_string", "model_status_string", "run_time" });
	auto const& info = highs.getInfo();
	out[0]["valid"] = factory.createScalar(info.valid);
	out[0]["mip_node_count"] = factory.createScalar(info.mip_node_count);
	out[0]["simplex_iteration_count"] = factory.createScalar(info.simplex_iteration_count);
	out[0]["ipm_iteration_count"] = factory.createScalar(info.ipm_iteration_count);
	out[0]["crossover_iteration_count"] = factory.createScalar(info.crossover_iteration_count);
	out[0]["pdlp_iteration_count"] = factory.createScalar(info.pdlp_iteration_count);
	out[0]["qp_iteration_count"] = factory.createScalar(info.qp_iteration_count);
	out[0]["primal_solution_status"] = factory.createScalar(info.primal_solution_status);
	out[0]["dual_solution_status"] = factory.createScalar(info.dual_solution_status);
	out[0]["basis_validity"] = factory.createScalar(info.basis_validity);
	out[0]["objective_function_value"] = factory.createScalar(info.objective_function_value);
	out[0]["mip_dual_bound"] = factory.createScalar(info.mip_dual_bound);
	out[0]["mip_gap"] = factory.createScalar(info.mip_gap);
	out[0]["max_integrality_violation"] = factory.createScalar(info.max_integrality_violation);
	out[0]["num_primal_infeasibilities"] = factory.createScalar(info.num_primal_infeasibilities);
	out[0]["max_primal_infeasibility"] = factory.createScalar(info.max_primal_infeasibility);
	out[0]["sum_primal_infeasibilities"] = factory.createScalar(info.sum_primal_infeasibilities);
	out[0]["num_dual_infeasibilities"] = factory.createScalar(info.num_dual_infeasibilities);
	out[0]["max_dual_infeasibility"] = factory.createScalar(info.max_dual_infeasibility);
	out[0]["sum_dual_infeasibilities"] = factory.createScalar(info.sum_dual_infeasibilities);
	out[0]["num_relative_primal_infeasibilities"] = factory.createScalar(info.num_relative_primal_infeasibilities);
	out[0]["max_relative_primal_infeasibility"] = factory.createScalar(info.max_relative_primal_infeasibility);
	out[0]["num_relative_dual_infeasibilities"] = factory.createScalar(info.num_relative_dual_infeasibilities);
	out[0]["max_relative_dual_infeasibility"] = factory.createScalar(info.max_relative_dual_infeasibility);
	out[0]["num_primal_residual_errors"] = factory.createScalar(info.num_primal_residual_errors);
	out[0]["max_primal_residual_error"] = factory.createScalar(info.max_primal_residual_error);
	out[0]["num_dual_residual_errors"] = factory.createScalar(info.num_dual_residual_errors);
	out[0]["max_dual_residual_error"] = factory.createScalar(info.max_dual_residual_error);
	out[0]["num_relative_primal_residual_errors"] = factory.createScalar(info.num_relative_primal_residual_errors);
	out[0]["max_relative_primal_residual_error"] = factory.createScalar(info.max_relative_primal_residual_error);
	out[0]["num_relative_dual_residual_errors"] = factory.createScalar(info.num_relative_dual_residual_errors);
	out[0]["max_relative_dual_residual_error"] = factory.createScalar(info.max_relative_dual_residual_error);
	out[0]["num_complementarity_violations"] = factory.createScalar(info.num_complementarity_violations);
	out[0]["max_complementarity_violation"] = factory.createScalar(info.max_complementarity_violation);
	out[0]["primal_dual_objective_error"] = factory.createScalar(info.primal_dual_objective_error);
	out[0]["primal_dual_integral"] = factory.createScalar(info.primal_dual_integral);
	// Extra fields
	out[0]["primal_solution_status_string"] = factory.createScalar(highs.solutionStatusToString(info.primal_solution_status));
	out[0]["dual_solution_status_string"] = factory.createScalar(highs.solutionStatusToString(info.dual_solution_status));
	out[0]["basis_validity_string"] = factory.createScalar(highs.basisValidityToString(info.basis_validity));
	out[0]["model_status_string"] = factory.createScalar(highs.modelStatusToString(highs.getModelStatus()));
	out[0]["run_time"] = factory.createScalar(highs.getRunTime());

	return out;
}


// Pre-condition: indx < numel(matStruct)
void matlabStructToHighsLinearObjective(HighsLinearObjective& out, const StructArray& matStruct, const size_t indx, const std::string& mexArgInNumberAsStr) {
	if (!isEqualFieldnames(linearObjectiveFields, getFieldNames(matStruct))) {
		throw std::runtime_error(std::format("Invalid linear objective struct array passed as {} input argument.", mexArgInNumberAsStr));
	}

	{
		throwIfInvalidFieldValue(matStruct, indx, "weight", ArrayType::DOUBLE, isScalar,
			std::format("Field \"weight\" of the linear objective struct at index {} of the {} input argument must be a scalar of double type.",
				indx + 1, mexArgInNumberAsStr)); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["weight"];
		out.weight = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "offset", ArrayType::DOUBLE, isScalar,
			std::format("Field \"offset\" of the linear objective struct at index {} of the {} input argument must be a scalar of double type.",
				indx + 1, mexArgInNumberAsStr)); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["offset"];
		out.offset = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "coefficients", ArrayType::DOUBLE, isVectorArr,
			std::format("Field \"coefficients\" of the linear objective struct at index {} of the {} input argument must be a vector of double type.",
				indx + 1, mexArgInNumberAsStr)); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["coefficients"];
		out.coefficients = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "abs_tolerance", ArrayType::DOUBLE, isScalar,
			std::format("Field \"abs_tolerance\" of the linear objective struct at index {} of the {} input argument must be a scalar of double type.",
				indx + 1, mexArgInNumberAsStr)); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["abs_tolerance"];
		out.abs_tolerance = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "rel_tolerance", ArrayType::DOUBLE, isScalar,
			std::format("Field \"rel_tolerance\" of the linear objective struct at index {} of the {} input argument must be a scalar of double type.",
				indx + 1, mexArgInNumberAsStr)); // Add 1 to the index to match MATLAB's indexing
		const TypedArray<double> arr = matStruct[indx]["rel_tolerance"];
		out.rel_tolerance = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, indx, "priority", HighsInt2MatlabArrayType, isScalar,
			std::format("Field \"priority\" of the linear objective struct at index {} of the {} input argument must be a scalar of {} type.",
				indx + 1, mexArgInNumberAsStr, HighsInt2MatlabClassStr)); // Add 1 to the index to match MATLAB's indexing
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
		throw std::runtime_error(std::format("Invalid solution struct passed as {} input argument.", mexArgInNumberAsStr));
	}

	HighsSolution out;
	{
		throwIfInvalidFieldValue(matStruct, 0, "value_valid", ArrayType::LOGICAL, isScalar,
			std::format("Field \"value_valid\" of the solution struct passed as the {} input argument must be a scalar of logical type.", mexArgInNumberAsStr));
		const TypedArray<bool> arr = matStruct[0]["value_valid"];
		out.value_valid = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "dual_valid", ArrayType::LOGICAL, isScalar,
			std::format("Field \"dual_valid\" of the solution struct passed as the {} input argument must be a scalar of logical type.", mexArgInNumberAsStr));
		const TypedArray<bool> arr = matStruct[0]["dual_valid"];
		out.dual_valid = arr[0];
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "col_value", ArrayType::DOUBLE, isVectorArr,
			std::format("Field \"col_value\" of the solution struct passed as the {} input argument must be a vector of double type.", mexArgInNumberAsStr));
		const TypedArray<double> arr = matStruct[0]["col_value"];
		out.col_value = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "col_dual", ArrayType::DOUBLE, isVectorArr,
			std::format("Field \"col_dual\" of the solution struct passed as the {} input argument must be a vector of double type.", mexArgInNumberAsStr));
		const TypedArray<double> arr = matStruct[0]["col_dual"];
		out.col_dual = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "row_value", ArrayType::DOUBLE, isVectorArr,
			std::format("Field \"row_value\" of the solution struct passed as the {} input argument must be a vector of double type.", mexArgInNumberAsStr));
		const TypedArray<double> arr = matStruct[0]["row_value"];
		out.row_value = matlabVectorToStdVector(arr);
	}
	{
		throwIfInvalidFieldValue(matStruct, 0, "row_dual", ArrayType::DOUBLE, isVectorArr,
			std::format("Field \"row_dual\" of the solution struct passed as the {} input argument must be a vector of double type.", mexArgInNumberAsStr));
		const TypedArray<double> arr = matStruct[0]["row_dual"];
		out.row_dual = matlabVectorToStdVector(arr);
	}
	return out;
}


/* Return the user settable options of HiGHS as a MATLAB struct.
* If getDefaults is true/false then the default/current values of the options are returned. */
StructArray highsOptionsToMatlabStruct(const Highs& highs, const bool getDefaults) {
	// Collect the names of all the user settable options
	std::vector<std::string> fieldnames(highs.getNumOptions());
	for (HighsInt i = 0; i < highs.getNumOptions(); ++i) {
		highs.getOptionName(i, &fieldnames[i]); // Note: This will always return kOk here
	}
	// Create ouput struct
	auto out = factory.createStructArray({ 1, 1 }, fieldnames);
	// Get values of all the user settable options
	for (auto const& fn : fieldnames) {
		HighsOptionType optType;
		highs.getOptionType(fn, &optType); // Note: This will always return kOk here
		switch (optType) {
		case HighsOptionType::kBool:
		{
			bool value;
			if (getDefaults) {
				highs.getBoolOptionValues(fn, nullptr, &value); // Note: This will always return kOk here
			}
			else {
				highs.getBoolOptionValues(fn, &value, nullptr); // Note: This will always return kOk here
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}

		case HighsOptionType::kInt:
		{
			HighsInt value;
			if (getDefaults) {
				highs.getIntOptionValues(fn, nullptr, nullptr, nullptr, &value); // Note: This will always return kOk here
			}
			else {
				highs.getIntOptionValues(fn, &value, nullptr, nullptr, nullptr); // Note: This will always return kOk here
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}

		case HighsOptionType::kDouble:
		{
			double value;
			if (getDefaults) {
				highs.getDoubleOptionValues(fn, nullptr, nullptr, nullptr, &value); // Note: This will always return kOk here
			}
			else {
				highs.getDoubleOptionValues(fn, &value, nullptr, nullptr, nullptr); // Note: This will always return kOk here
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}

		case HighsOptionType::kString:
		{
			std::string value;
			if (getDefaults) {
				highs.getStringOptionValues(fn, nullptr, &value); // Note: This will always return kOk here
			}
			else {
				highs.getStringOptionValues(fn, &value, nullptr); // Note: This will always return kOk here
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}
		}
	}
	return out;
}


// Set HiGHS options by taking values from a MATLAB struct
// Pre-condition: opts is a 1x1 struct
void setHighsOptions(Highs& highs, const StructArray& opts, const std::string& mexArgInNumberAsStr) {
	// Get the fieldnames of the input MATLAB struct
	auto const fieldnames = getFieldNames(opts);
	// Set HiGHS options
	for (auto const& fn : fieldnames) {
		HighsOptionType optType;
		if (highs.getOptionType(fn, &optType) != HighsStatus::kOk) {
			throw std::runtime_error(std::format("Invalid option provided in the struct passed as the {} input argument. \"{}\" is not a legal HiGHS option.",
				mexArgInNumberAsStr, fn));
		}
		switch (optType) {
		case HighsOptionType::kBool:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::LOGICAL, isScalar,
				std::format("Field \"{}\" of the options struct passed as the {} input argument must be a scalar of logical type.",
					fn, mexArgInNumberAsStr));
			const TypedArray<bool> value = opts[0][fn];
			if (highs.setOptionValue(fn, static_cast<bool>(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(std::format("Failed to set the HiGHS option \"{}\". The option struct was passed as the {} input argument.",
					fn, mexArgInNumberAsStr));
			}
			break;
		}

		case HighsOptionType::kInt:
		{
			throwIfInvalidFieldValue(opts, 0, fn, HighsInt2MatlabArrayType, isScalar,
				std::format("Field \"{}\" of the options struct passed as the {} input argument must be a scalar of {} type.",
					fn, mexArgInNumberAsStr, HighsInt2MatlabClassStr));
			const TypedArray<HighsInt> value = opts[0][fn];
			if (highs.setOptionValue(fn, castToHighsInt(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(std::format("Failed to set the HiGHS option \"{}\". The option struct was passed as the {} input argument.",
					fn, mexArgInNumberAsStr));
			}
			break;
		}

		case HighsOptionType::kDouble:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::DOUBLE, isScalar,
				std::format("Field \"{}\" of the options struct passed as the {} input argument must be a scalar of double type.",
					fn, mexArgInNumberAsStr));
			const TypedArray<double> value = opts[0][fn];
			if (highs.setOptionValue(fn, static_cast<double>(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(std::format("Failed to set the HiGHS option \"{}\". The option struct was passed as the {} input argument.",
					fn, mexArgInNumberAsStr));
			}
			break;
		}

		case HighsOptionType::kString:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::MATLAB_STRING, isScalar,
				std::format("Field \"{}\" of the options struct passed as the {} input argument must be a MATLAB string.",
					fn, mexArgInNumberAsStr));
			const TypedArray<MATLABString> value = opts[0][fn];
			if (highs.setOptionValue(fn, matlabStringToStdString(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(std::format("Failed to set the HiGHS option \"{}\". The option struct was passed as the {} input argument.",
					fn, mexArgInNumberAsStr));
			}
			break;
		}
		}

	}
}


//// Convert MATLAB full matrix to HIGHS sparse representation. This implementation is slow because accessing MATLAB TypedArray elements using operator[] is slow.
//void matlabMatrixToHighsFormat(
//	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
//	const TypedArray<double>& A, const HighsInt nrow, const HighsInt ncol, const bool doTril // inputs
//) {
//	// Count the number of non-zero elements
//	HighsInt nnz = 0;
//	for (HighsInt j = 0; j < ncol; ++j) {
//		for (HighsInt i = doTril ? j : 0; i < nrow; ++i) {
//			if (A[i][j] != 0) ++nnz;
//		}
//	}
//	// Resize outputs
//	start.resize(ncol + 1);
//	index.resize(nnz);
//	value.resize(nnz);
//	// Loop over all (or lower triangular) the elements of A and copy non-zero values to the outputs	
//	HighsInt k = 0;
//	for (HighsInt j = 0; j < ncol; ++j) {
//		start[j] = k;
//		for (HighsInt i = doTril ? j : 0; i < nrow; ++i) {
//			if (!A[i][j]) continue;
//			index[k] = i;
//			value[k] = A[i][j];
//			++k;
//		}
//	}
//	// Here k == nnz
//	start[ncol] = nnz;
//}
// Convert MATLAB full matrix to HIGHS sparse representation.
HighsInt matlabMatrixToHighsFormat(
	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
	const TypedArray<double>& A, const HighsInt nrow, const HighsInt ncol, const bool doTril // inputs
) {
	if (A.getMemoryLayout() != MemoryLayout::COLUMN_MAJOR) {
		throw std::runtime_error("Input matrix must be in column major order.");
	}
	auto pA = getPointer(A);
	// Count the number of non-zero elements
	auto pAcol = pA; // Pointer to the first element of the first column of A
	HighsInt nnz = 0;
	for (HighsInt j = 0; j < ncol; ++j) {
		for (HighsInt i = doTril ? j : 0; i < nrow; ++i) {
			if (pAcol[i] != 0) ++nnz;
		}
		pAcol += nrow; // Move to the next column
	}
	// Resize outputs
	start.resize(ncol + 1);
	index.resize(nnz);
	value.resize(nnz);
	// Loop over all (or lower triangular) the elements of A and copy non-zero values to the outputs	
	if (!nnz) {
		std::fill(start.begin(), start.end(), 0);
	}
	else {
		pAcol = pA; // Pointer to the first element of the first column of A
		HighsInt k = 0;
		for (HighsInt j = 0; j < ncol; ++j) {
			start[j] = k;
			for (HighsInt i = doTril ? j : 0; i < nrow; ++i) {
				if (!pAcol[i]) continue;
				index[k] = i;
				value[k] = pAcol[i];
				++k;
			}
			pAcol += nrow; // Move to the next column
		}
		// Here k == nnz
		start[ncol] = nnz;
	}
	return nnz;
}


//// Convert MATLAB sparse matrix to HIGHS sparse representation. This implementation is slow because accessing MATLAB SparseArray elements using iterators is slow.
//void matlabMatrixToHighsFormat(
//	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
//	const SparseArray<double>& A, const HighsInt, const HighsInt ncol, const bool doTril // inputs
//) {
//	if (A.getMemoryLayout() != MemoryLayout::COLUMN_MAJOR) {
//		throw std::runtime_error("Input sparse matrix must be in column major order."); // We need this because we want the SparseArray iterator to iterate in column major order
//	}
//	// Count the number of non-zero elements
//	HighsInt nnz = 0;
//	if (doTril) {
//		for (auto end = A.end(), it = A.begin(); it != end; ++it) {
//			auto const inz = A.getIndex(it);
//			if (inz.first < inz.second) continue; // Skip strictly upper-triangular elements of A
//			++nnz;
//		}
//	}
//	else {
//		nnz = castToHighsInt(A.getNumberOfNonZeroElements());
//	}
//	// Resize outputs
//	start.resize(ncol + 1);
//	index.resize(nnz);
//	value.resize(nnz);
//	// Loop over all (or lower triangular) and non-zero elements of A and copy the values to the outputs
//	HighsInt k = 0;
//	std::vector<HighsInt> nnzCol(ncol); // nnzCol[i] is the number of non-zero elements in the i'th column of A
//	nnzCol.assign(ncol, 0);
//	for (auto end = A.end(), it = A.begin(); it != end; ++it) {
//		auto const inz = A.getIndex(it); // inz.first/.second is the row/column index of the (non-zero) element pointed to by iterator it.
//		if (doTril && inz.first < inz.second) continue; // Skip strictly upper-triangular elements of A
//		++nnzCol[inz.second];
//		index[k] = castToHighsInt(inz.first);
//		value[k] = *it;
//		++k;
//	}
//	// Set start
//	start[0] = 0;
//	for (HighsInt j = 1; j <= ncol; ++j) {
//		start[j] = start[j - 1] + nnzCol[j - 1];
//	}
//}
// Convert MATLAB sparse matrix to HIGHS sparse representation. The sparse matrix must be specified by the triplet iA, jA, and, vA, where, [iA, jA, vA]=find(A).
// Pre-condition: iA, jA, vA are vectors of the same length and represent the row indices, column indices (MATLAB based i.e. starting at 1) and values of the non-zero elements of the sparse matrix A respectively.
HighsInt matlabMatrixToHighsFormat(
	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
	const TypedArray<double>& iA, const TypedArray<double>& jA, const TypedArray<double>& vA, const HighsInt, const HighsInt ncol, const bool doTril // inputs
) {
	auto pI = getPointer(iA), pJ = getPointer(jA), pV = getPointer(vA); // Pointers to the first elements of i, j, v respectively
	const size_t nA = numel(iA);
	// Count the number of non-zero elements
	HighsInt nnz = 0;
	if (doTril) {
		for (size_t i = 0; i < nA; ++i) {
			if (pI[i] < pJ[i]) continue; // Skip strictly upper-triangular elements of A
			++nnz;
		}
	}
	else {
		nnz = castToHighsInt(nA);
	}
	// Resize outputs
	start.resize(ncol + 1);
	index.resize(nnz);
	value.resize(nnz);
	// Loop over all (or lower triangular) and non-zero elements of A and copy the values to the outputs
	HighsInt k = 0;
	std::vector<HighsInt> nnzCol(ncol); // nnzCol[i] is the number of non-zero elements in the i'th column of A
	nnzCol.assign(ncol, 0);
	for (size_t i = 0; i < nA; ++i) {
		if (doTril && pI[i] < pJ[i]) continue; // Skip strictly upper-triangular elements of A
		++nnzCol[castToHighsInt(pJ[i] - 1)]; // -1 to convert MATLAB (one) based index to C++ (zero) based index
		index[k] = castToHighsInt(pI[i] - 1); // -1 to convert MATLAB (one) based index to C++ (zero) based index
		value[k] = pV[i];
		++k;
	}
	// Set start
	start[0] = 0;
	for (HighsInt j = 1; j <= ncol; ++j) {
		start[j] = start[j - 1] + nnzCol[j - 1];
	}
	return nnz;
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


class MexFunction : public Function {

	std::shared_ptr<MATLABEngine> mtlbEngPtr = getEngine();
	// Error messages passed by HiGHS inside the logging callback are stored here
	std::stack<std::string> highsLogErrStack;


	std::string getFunctionNameString() {
		return convertUTF16StringToUTF8String(getFunctionName());
	}

	// Display std::vector on the MATLAB console. This method is used while debugging.
	template <typename T>
	void disp(const std::vector<T>& v) {
		mtlbEngPtr->feval(u"disp", 0, std::vector<Array>({ stdVectorToMatlabVector(v, true) }));
	}

	// Display Array on the MATLAB console. This method is used while debugging.
	void disp(const Array& arr) {
		mtlbEngPtr->feval(u"disp", 0, std::vector<Array>({ arr }));
	}

	// Display string on the MATLAB console.
	void print(const std::string& msg) {
		mtlbEngPtr->feval(u"fprintf", 0, std::vector<Array>(
			{ factory.createScalar("%s"), factory.createScalar(msg) }));
	}

	// Display warning message.
	void warning(const std::string& msg) {
		auto const str = std::format("in mex function {}: {}\n", getFunctionNameString(), msg);
		mtlbEngPtr->feval(u"warning", 0, std::vector<Array>({
			factory.createScalar("highs:mex"), factory.createScalar(str)
			}));
	}

	// Display error message. This method is meant to be called inside the catch blocks of operator()(...) method.
	void error__(const std::string& msg) {
		auto const str = std::format("Error in mex function {}:\n{}\n", getFunctionNameString(), msg);
		mtlbEngPtr->feval(u"error", 0, std::vector<Array>({ factory.createScalar(str) }));
	}

	// Callback of to log HiGHS messages to the MATLAB console
	// Pre-condition: callbackType should always be HighsCallbackType::kCallbackLogging. 
	//                Hence, make sure to call Highs::startCallback(...) method with 
	//                HighsCallbackType::kCallbackLogging only as input. 
	void logCallback(const int callbackType, const HighsLogType logType, const std::string& message) {
		switch (logType) {
		case HighsLogType::kInfo:
			print(message);
			break;

		case HighsLogType::kDetailed:
			print(message);
			break;

		case HighsLogType::kVerbose:
			print(message);
			break;

		case HighsLogType::kWarning:
			warning(message);
			break;

		case HighsLogType::kError:
			highsLogErrStack.push(message);
			break;
		}
	}

	void throwIfHighsError() {
		if (highsLogErrStack.empty()) return; // No error occured
		std::string msg;
		while (!highsLogErrStack.empty()) {
			msg += highsLogErrStack.top();
			highsLogErrStack.pop();
		}
		throw std::runtime_error(msg);
	}

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

	void checkHighsReturnStatus(const HighsStatus status, const std::string& warnMsg, const std::string& errMsg) {
		// Throw an error if HiGHS passed the error message via the logging callback
		throwIfHighsError();
		// Check for the return status
		switch (status) {
		case HighsStatus::kError:
			throw std::runtime_error(errMsg);
			break;

		case HighsStatus::kWarning:
			warning(warnMsg);
			break;

		default:
			; // Do nothing
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
				throw std::runtime_error(std::format("Second input argument (A) must be a matrix of double type with {} columns.", highsModel.lp_.num_col_));
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
				throw std::runtime_error(std::format("The 5th element of the cell array passed as the second input argument (A) must be a double scalar equal to {}.", highsModel.lp_.num_col_));
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
				throw std::runtime_error(std::format("Expected length of third input argument (L) to be {}.", highsModel.lp_.num_row_));
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
				throw std::runtime_error(std::format("Expected length of fourth input argument (U) to be {}.", highsModel.lp_.num_row_));
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
				throw std::runtime_error(std::format("Expected length of fifth input argument (l) to be {}.", highsModel.lp_.num_col_));
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
				throw std::runtime_error(std::format("Expected length of sixth input argument (u) to be {}.", highsModel.lp_.num_col_));
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
				throw std::runtime_error(std::format("Expected dimension of the seventh input argument (Q) to be {}.", highsModel.lp_.num_col_));
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
				throw std::runtime_error(std::format("Expected length of the eighth input argument (integrality) to be {}.", highsModel.lp_.num_col_));
			}
			highsModel.lp_.integrality_.resize(numel(integralityStrings));
			for (size_t i = 0; i < numel(integralityStrings); ++i) {
				auto const integralityStr = matlabStringToStdString(integralityStrings[i]);
				auto const it = integralityStringsMap.find(integralityStr);
				if (it == integralityStringsMap.end()) {
					throw std::runtime_error(std::format("Invalid string at index {} of the eighth input argument (integrality). \"{}\" is not a valid integrality string.",
						i + 1, integralityStr)); // Add 1 to the index to match MATLAB's indexing
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
			print(std::format("lp_.sense_ = {}\n", highsModel.lp_.sense_ == ObjSense::kMinimize ? "kMinimize" : "kMaximize"));
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
					throw std::runtime_error(std::format("Expected length of the eleventh input argument (setSoln) to be {}.", numCol));
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
