/* Code shared by the mex wrappers over the HiGHS optimization library
 * (https://github.com/ERGO-Code/HiGHS)
 *
 * This header holds everything that highsmex.cpp and highsmex_iis.cpp have in
 * common: the MATLAB type helpers, the conversion of HiGHS options and info to
 * MATLAB structs, the conversion of MATLAB matrices to the HiGHS sparse format,
 * and the mex infrastructure (logging, warnings, errors) in the form of the
 * HighsMexBase class.
 *
 * It is header-only, so the mex build is a single command per mex file, exactly
 * as before. Free functions are inline; `factory` is an inline variable, which
 * requires C++17 and is therefore covered by the C++20 switch that both mex
 * files are compiled with.
 *
 * NOTE: mexAdapter.hpp must NOT be included here. It defines the entry point of
 *       a mex file and expects a class named MexFunction, so it belongs in
 *       highsmex.cpp and highsmex_iis.cpp, one per mex file.
 *
 * Author: Savyasachi Singh
 *
 * Covered by the MIT License (see LICENSE file for details).
 * See https://github.com/savyasachi/HiGHSMEX for more information.
 */

#ifndef HIGHSMEX_COMMON_HPP
#define HIGHSMEX_COMMON_HPP


 // Include standard headers
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <sstream>
#include <functional>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <complex>
// Include C++ MEX API. Note that mexAdapter.hpp is deliberately not included, see above.
#include "mex.hpp"
#include "MatlabDataArray.hpp"
// Include HiGHS
#include "Highs.h"


// Open MATLAB namespaces. These are at global scope in both mex sources, so that
// behaviour is kept by putting them here.
using namespace matlab::mex;
using namespace matlab::data;
using namespace matlab::engine;


/* ------------------------------------------------------------------------------------------------------ */
/*                                         VARIABLES                                                      */
/* ------------------------------------------------------------------------------------------------------ */

// Compile time constants
constexpr ArrayType   HighsInt2MatlabArrayType = std::is_same_v<HighsInt, int32_t> ? ArrayType::INT32 : ArrayType::INT64;
const std::string HighsInt2MatlabClassStr = std::is_same_v<HighsInt, int32_t> ? "int32" : "int64";
constexpr bool        MexDebugPrinting = 0;

// Variables
inline ArrayFactory factory;


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
inline bool isEqualFieldnames(const std::vector<std::string>& f1, const std::vector<std::string>& f2) {
	if (f1.size() != f2.size()) return false;
	const std::set<std::string> f2set(f2.begin(), f2.end());
	for (auto const& s : f1) {
		if (f2set.find(s) == f2set.end()) return false;
	}
	return true;
}


// Convert MATLAB string to std::string.
inline std::string matlabStringToStdString(const MATLABString& matlabStr) {
	return matlabStr.has_value() ? convertUTF16StringToUTF8String(*matlabStr) : "";
}

template <typename T>
void appendToStream(std::ostringstream& oss, const T& value) {
	oss << value;
}

inline void appendToStream(std::ostringstream& oss, const std::string& value) {
	oss << value;
}

inline void appendToStream(std::ostringstream& oss, const char* value) {
	oss << value;
}

inline void appendToStream(std::ostringstream& oss, const MATLABString& value) {
	oss << matlabStringToStdString(value);
}

template <typename... Args>
std::string formatMessage(const Args&... args) {
	std::ostringstream oss;
	(appendToStream(oss, args), ...);
	return oss.str();
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


// Convert HighsInfo to MATLAB struct. We add some extra fields to the output MATLAB struct.
inline StructArray highsInfoToMatlabStruct(const Highs& highs) {
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
		// These fields are extra. They are added by HiGHSMEX.
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


/* Return the user settable options of HiGHS as a MATLAB struct.
* If getDefaults is true/false then the default/current values of the options are returned. */
inline StructArray highsOptionsToMatlabStruct(const Highs& highs, const bool getDefaults) {
	// Collect the names of all the user settable options
	auto const numOptions = highs.getNumOptions();
	std::vector<std::string> fieldnames(numOptions);
	for (HighsInt i = 0; i < numOptions; ++i) {
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
inline void setHighsOptions(Highs& highs, const StructArray& opts, const std::string& mexArgInNumberAsStr) {
	// Get the fieldnames of the input MATLAB struct
	auto const fieldnames = getFieldNames(opts);
	// Set HiGHS options
	for (auto const& fn : fieldnames) {
		HighsOptionType optType;
		if (highs.getOptionType(fn, &optType) != HighsStatus::kOk) {
			throw std::runtime_error(formatMessage("Invalid option provided in the struct passed as the ", mexArgInNumberAsStr, " input argument. \"", fn, "\" is not a legal HiGHS option."));
		}
		switch (optType) {
		case HighsOptionType::kBool:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::LOGICAL, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of logical type."));
			const TypedArray<bool> value = opts[0][fn];
			if (highs.setOptionValue(fn, static_cast<bool>(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
			}
			break;
		}

		case HighsOptionType::kInt:
		{
			throwIfInvalidFieldValue(opts, 0, fn, HighsInt2MatlabArrayType, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of ", HighsInt2MatlabClassStr, " type."));
			const TypedArray<HighsInt> value = opts[0][fn];
			if (highs.setOptionValue(fn, castToHighsInt(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
			}
			break;
		}

		case HighsOptionType::kDouble:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::DOUBLE, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of double type."));
			const TypedArray<double> value = opts[0][fn];
			if (highs.setOptionValue(fn, static_cast<double>(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
			}
			break;
		}

		case HighsOptionType::kString:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::MATLAB_STRING, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a MATLAB string."));
			const TypedArray<MATLABString> value = opts[0][fn];
			if (highs.setOptionValue(fn, matlabStringToStdString(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
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
/* Convert MATLAB full matrix to HIGHS sparse representation.
 * doTril selects the lower triangle only, which is what HiGHS wants for the
 * Hessian of a quadratic objective. It defaults to false, i.e. the whole matrix.
 */
inline HighsInt matlabMatrixToHighsFormat(
	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
	const TypedArray<double>& A, const HighsInt nrow, const HighsInt ncol, const bool doTril = false // inputs
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
/* Convert MATLAB sparse matrix to HIGHS sparse representation. The sparse matrix must be specified by the triplet iA, jA, and, vA, where, [iA, jA, vA]=find(A).
 * Pre-condition: iA, jA, vA are vectors of the same length and represent the row indices, column indices (MATLAB based i.e. starting at 1) and values of the non-zero elements of the sparse matrix A respectively.
 * doTril selects the lower triangle only, see the overload above.
 */
inline HighsInt matlabMatrixToHighsFormat(
	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
	const TypedArray<double>& iA, const TypedArray<double>& jA, const TypedArray<double>& vA, const HighsInt, const HighsInt ncol, const bool doTril = false // inputs
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
/*                                       MEX INFRASTRUCTURE                                               */
/* ------------------------------------------------------------------------------------------------------ */

/* Base class of the MexFunction classes of highsmex and highsmex_iis. It holds
 * the connection to the MATLAB engine and everything that reports back to
 * MATLAB: console output, warnings, errors, and the HiGHS logging callback.
 *
 * The derived class must be named MexFunction, because that is the name that
 * mexAdapter.hpp instantiates.
 */
class HighsMexBase : public Function {

protected:

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
		auto const str = formatMessage("in mex function ", getFunctionNameString(), ": ", msg, "\n");
		mtlbEngPtr->feval(u"warning", 0, std::vector<Array>({
			factory.createScalar("highs:mex"), factory.createScalar(str)
			}));
	}

	// Display error message. This method is meant to be called inside the catch blocks of operator()(...) method.
	void error__(const std::string& msg) {
		auto const str = formatMessage("Error in mex function ", getFunctionNameString(), ":\n", msg, "\n");
		mtlbEngPtr->feval(u"error", 0, std::vector<Array>({ factory.createScalar(str) }));
	}

	// Callback to log HiGHS messages to the MATLAB console
	// Pre-condition: callbackType should always be HighsCallbackType::kCallbackLogging.
	//                Hence, make sure to call Highs::startCallback(...) method with
	//                HighsCallbackType::kCallbackLogging only as input.
	void logCallback(const int, const HighsLogType logType, const std::string& message) {
		switch (logType) {
		case HighsLogType::kInfo:
		case HighsLogType::kDetailed:
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

};

#endif // HIGHSMEX_COMMON_HPP
// EOF
