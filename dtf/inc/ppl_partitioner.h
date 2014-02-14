#define _CONCRT_ASSERT _ASSERTE
namespace Concurrency
{
namespace details
{

	//
    // _MallocaArrayHolder is used when the allocation size is known up front, and the memory must be allocated in a contiguous space
    //
    template<typename _ElemType>
    class _MallocaArrayHolder
    {
    public:

        _MallocaArrayHolder() : _M_ElemArray(NULL), _M_ElemsConstructed(0) {}

        // _Initialize takes the pointer to the memory allocated by the user via _malloca
        void _Initialize(_ElemType * _Elem)
        {
            // The object must be initialized exactly once
            _CONCRT_ASSERT(_M_ElemArray == NULL && _M_ElemsConstructed == 0);
            _M_ElemArray = _Elem;
            _M_ElemsConstructed = 0;
        }

        // Register the next slot for destruction. Because we only keep the index of the last slot to be destructed,
        // this method must be called sequentially from 0 to N where N < _ElemCount.
        void _IncrementConstructedElemsCount()
        {
            _CONCRT_ASSERT(_M_ElemArray != NULL); // must already be initialized
            _M_ElemsConstructed++;
        }

        virtual ~_MallocaArrayHolder()
        {
            for( size_t _I=0; _I < _M_ElemsConstructed; ++_I )
            {
                _M_ElemArray[_I]._ElemType::~_ElemType();
            }
            // Works even when object was not initialized, i.e. _M_ElemArray == NULL
            _freea(_M_ElemArray);
        }
    private:
        _ElemType * _M_ElemArray;
        size_t     _M_ElemsConstructed;

        // not supposed to be copy-constructed or assigned
        _MallocaArrayHolder(const _MallocaArrayHolder & );
        _MallocaArrayHolder&  operator = (const _MallocaArrayHolder & );
    };

	    //
    // _MallocaListHolder is used when the allocation size is not known up front, and the elements are added to the list dynamically
    //
    template<typename _ElemType>
    class _MallocaListHolder
    {
    public:
        // Returns the size required to allocate the payload itself and the pointer to the next element
        size_t _GetAllocationSize() const
        {
            return sizeof(_ElemNodeType);
        }

        _MallocaListHolder() : _M_FirstNode(NULL) 
        {
        }

        // Add the next element to the list. The memory is allocated in the caller's frame by _malloca
        void _AddNode(_ElemType * _Elem)
        {
            _ElemNodeType * _Node = reinterpret_cast<_ElemNodeType *>(_Elem);
            _Node->_M_Next = _M_FirstNode;
            _M_FirstNode = reinterpret_cast<_ElemNodeType *>(_Elem);
        }

        // Walk the list and destruct, then free each element
        virtual ~_MallocaListHolder()
        {
            for( _ElemNodeType * _Node = _M_FirstNode; _Node != NULL; )
            {
                _ElemNodeType* _M_Next = _Node->_M_Next;
                _Node->_M_Elem._ElemType::~_ElemType();
                 _freea(_Node);
                _Node = _M_Next;
            }
        }

    private:

        class _ElemNodeType
        {
            friend class _MallocaListHolder;
            _ElemType _M_Elem;
            _ElemNodeType * _M_Next;
            // always instatiated via malloc, so default ctor and dtor not needed
            _ElemNodeType();
            ~_ElemNodeType();
            // not supposed to be copy-constructed or assigned
            _ElemNodeType(const _ElemNodeType & );
            _ElemNodeType &  operator = (const _ElemNodeType & );
        };

        _ElemNodeType* _M_FirstNode;

        // not supposed to be copy-constructed or assigned
        _MallocaListHolder(const _MallocaListHolder & );
        _MallocaListHolder &  operator = (const _MallocaListHolder & );
    };

}

/// <summary>
///     The <c>default_partitioner</c> class represents the default method <c>parallel_for</c>, <c>parallel_for_each</c> and 
///     <c>parallel_transform</c> use to partition the range they iterates over. This method of partitioning employes range stealing
///     for load balancing as well as per-iterate cancellation.
/// </summary>
/**/
class default_partitioner
{
public:
    /// <summary>
    ///     Constructs a <c>default_partitioner</c> object.
    /// </summary>
    /**/
    default_partitioner() {}

    /// <summary>
    ///     Destroys a <c>default_partitioner</c> object.
    /// </summary>
    /**/
    ~default_partitioner() {}

    unsigned int _Get_num_chunks() const
    {
        return CurrentScheduler::Get()->GetNumberOfVirtualProcessors();
    }
};


/// <summary>
///     The <c>fixed_partitioner</c> class represents a static partitioning of the range iterated over by <c>parallel_for</c>. The partitioner
///     divides the range into as many chunks as there are workers available to the underyling scheduler.
/// </summary>
/**/
class fixed_partitioner
{
public:
    /// <summary>
    ///     Constructs a <c>fixed_partitioner</c> object.
    /// </summary>
    /**/
    fixed_partitioner(int num = 0) : _M_num_chunks(num)
	{
		if (num ==0)
		{
			_M_num_chunks= CurrentScheduler::Get()->GetNumberOfVirtualProcessors();
		}
	}

    /// <summary>
    ///     Destroys a <c>fixed_partitioner</c> object.
    /// </summary>
    /**/
    ~fixed_partitioner() {}
	
    unsigned int _Get_num_chunks() const
    {
        return _M_num_chunks;
    }

private:
	int _M_num_chunks;
};
template <typename _Random_iterator, typename _Index_type, typename _Function, typename _Partitioner, bool _Is_iterator>
class _Parallel_fixed_chunk_helper
{
public:
    _Parallel_fixed_chunk_helper(_Index_type, const _Random_iterator& _First, _Index_type _First_iteration,
         _Index_type _Last_iteration, const _Index_type& _Step, const _Function& _Func, const fixed_partitioner&) :
        _M_first(_First), _M_first_iteration(_First_iteration), _M_last_iteration(_Last_iteration), _M_step(_Step), _M_function(_Func)
    {
        // Empty constructor since members are already assigned
    }

    __declspec(safebuffers) void operator()() const
    {
        // Keep the secondary, scaled, loop index for quick indexing into the data structure
        _Index_type _Scaled_index = _M_first_iteration * _M_step;

        for (_Index_type _I = _M_first_iteration; _I < _M_last_iteration; (_I++, _Scaled_index += _M_step))
        {
            // Execute one iteration: the element is at scaled index away from the first element.
            _Parallel_chunk_helper_invoke<_Random_iterator, _Index_type, _Function, _Is_iterator>::_Invoke(_M_first, _Scaled_index, _M_function);
        }
    }
private:

    const _Random_iterator&            _M_first;
    const _Index_type&                 _M_step;
    const _Function&                   _M_function;

    const _Index_type                  _M_first_iteration;
    const _Index_type                  _M_last_iteration;

    _Parallel_fixed_chunk_helper const & operator=(_Parallel_fixed_chunk_helper const&);    // no assignment operator
};

template <typename _Worker_class, typename _Index_type, typename Partitioner>
void _Parallel_chunk_task_group_run(structured_task_group& _Task_group,
                                    task_handle<_Worker_class>* _Chunk_helpers,
                                    const Partitioner&,
                                    _Index_type _I)
{
    _Task_group.run(_Chunk_helpers[_I]);
}

// Helper functions that implement parallel_for

template <typename _Worker_class, typename _Random_iterator, typename _Index_type, typename _Function, typename _Partitioner>
void _Parallel_chunk_impl(const _Random_iterator& _First, _Index_type _Range, const _Index_type& _Step, const _Function& _Func, _Partitioner&& _Part)
{
    _CONCRT_ASSERT(_Range > 1);
    _CONCRT_ASSERT(_Step > 0);

    _Index_type _Num_chunks = static_cast<_Index_type>(_Part._Get_num_chunks());
    _CONCRT_ASSERT(_Num_chunks > 0);

    _Index_type _Num_iterations = (_Step == 1) ? _Range : (((_Range - 1) / _Step) + 1);
    _CONCRT_ASSERT(_Num_iterations > 1);

    // Allocate memory on the stack for task_handles to ensure everything is properly structured.
    task_handle<_Worker_class> * _Chunk_helpers = (task_handle<_Worker_class> *) _malloca(sizeof(task_handle<_Worker_class>) * static_cast<size_t>(_Num_chunks));
    ::Concurrency::details::_MallocaArrayHolder<task_handle<_Worker_class>> _Holder;
    _Holder._Initialize(_Chunk_helpers);

    structured_task_group _Task_group;

    _Index_type _Iterations_per_chunk = _Num_iterations / _Num_chunks;
    _Index_type _Remaining_iterations = _Num_iterations % _Num_chunks;

    // If there are less iterations than desired chunks, set the chunk number
    // to be the number of iterations.
    if (_Iterations_per_chunk == 0)
    {
        _Num_chunks = _Remaining_iterations;
    }

    _Index_type _Work_size = 0;
    _Index_type _Start_iteration = 0;
    _Index_type _I;

    // Split the available work in chunks
    for (_I = 0; _I < _Num_chunks - 1; _I++)
    {
        if (_Remaining_iterations > 0)
        {
            // Iterations are not divided evenly, so add 1 remainder iteration each time
            _Work_size = _Iterations_per_chunk + 1;
            _Remaining_iterations--;
        }
        else
        {
            _Work_size = _Iterations_per_chunk;
        }

        // New up a task_handle "in-place", in the array preallocated on the stack
        new(&_Chunk_helpers[_I]) task_handle<_Worker_class>(_Worker_class(_I, _First, _Start_iteration, _Start_iteration + _Work_size, _Step, _Func, std::forward<_Partitioner>(_Part)));
        _Holder._IncrementConstructedElemsCount();

        // Run each of the chunk tasks in parallel
        _Parallel_chunk_task_group_run(_Task_group, _Chunk_helpers, std::forward<_Partitioner>(_Part), _I);

        // Prepare for the next iteration
        _Start_iteration += _Work_size;
    }

    // Since this is the last iteration, then work size might be different
    _CONCRT_ASSERT((_Remaining_iterations == 0) || ((_Iterations_per_chunk == 0) && (_Remaining_iterations == 1)));
    _Work_size = _Num_iterations - _Start_iteration;

    // New up a task_handle "in-place", in the array preallocated on the stack
    new(&_Chunk_helpers[_I]) task_handle<_Worker_class>(_Worker_class(_I, _First, _Start_iteration, _Start_iteration + _Work_size, _Step, _Func, std::forward<_Partitioner>(_Part)));
    _Holder._IncrementConstructedElemsCount();

    _Task_group.run_and_wait(_Chunk_helpers[_I]);
}

// Helper for the parallel_for API with a fixed partitioner - creates a fixed number of chunks up front with no range-stealing enabled.
template <typename _Index_type, typename _Diff_type, typename _Function>
__declspec(safebuffers) void _Parallel_for_partitioned_impl(_Index_type _First, _Diff_type _Range, _Diff_type _Step, const _Function& _Func, const fixed_partitioner& _Part)
{
    typedef _Parallel_fixed_chunk_helper<_Index_type, _Diff_type, _Function, fixed_partitioner, false> _Worker_class;
    _Parallel_chunk_impl<_Worker_class>(_First, _Range, _Step, _Func, _Part);
}


template <typename _Index_type, typename _Function, typename _Partitioner>
__declspec(safebuffers) void _Parallel_for_impl(_Index_type _First, _Index_type _Last, _Index_type _Step, const _Function& _Func, _Partitioner&& _Part)
{
    // The step argument must be 1 or greater; otherwise it is an invalid argument
    if (_Step < 1)
    {
        throw std::invalid_argument("_Step");
    }

    // If there are no elements in this range we just return
    if (_First >= _Last)
    {
        return;
    }

    // Compute the difference type based on the arguments and avoid signed overflow for int, long, and long long
    typedef typename std::tr1::conditional<std::tr1::is_same<_Index_type, int>::value, unsigned int,
        typename std::tr1::conditional<std::tr1::is_same<_Index_type, long>::value, unsigned long,
            typename std::tr1::conditional<std::tr1::is_same<_Index_type, long long>::value, unsigned long long, decltype(_Last - _First)
            >::type
        >::type
    >::type _Diff_type;

    _Diff_type _Range = _Last - _First;
    _Diff_type _Diff_step = _Step;

    if (_Range <= _Diff_step)
    {
        _Func(_First);
    }
    else
    {
        _Parallel_for_partitioned_impl<_Index_type, _Diff_type, _Function>(_First, _Range, _Step, _Func, std::forward<_Partitioner>(_Part));
    }
}


// <summary>
///     <c>parallel_for</c> iterates over a range of indices and executes a user-supplied function at each iteration, in parallel.
/// </summary>
/// <typeparam name="_Index_type">
///     The type of the index being used for the iteration.
/// </typeparam>
/// <typeparam name="_Function">
///     The type of the function that will be executed at each iteration.
/// </typeparam>
/// <typeparam name="_Partitioner">
///     The type of the partitioner that is used to partition the supplied range.
/// </typeparam>
/// <param name="_First">
///     The first index to be included in the iteration.
/// </param>
/// <param name="_Last">
///     The index one past the last index to be included in the iteration.
/// </param>
/// <param name="_Step">
///     The value by which to step when iterating from <paramref name="_First"/> to <paramref name="_Last"/>. The step must be positive.
///     <see cref="invalid_argument Class">invalid_argument</see> is thrown if the step is less than 1.
/// </param>
/// <param name="_Func">
///     The function to be executed at each iteration. This may be a lambda expression, a function pointer, or any object
///     that supports a version of the function call operator with the signature
///     <c>void operator()(</c><typeparamref name="_Index_type"/><c>)</c>.
/// </param>
/// <param name="_Part">
///     A reference to the partitioner object. The argument can be one of <c>const</c> <see ref="default_partitioner Class">default_partitioner</see><c>&amp;</c>,
///     <c>const</c> <see ref="fixed_partitioner Class">fixed_partitioner</see><c>&amp;</c> or <see ref="affinity_partitioner Class">affinity_partitioner</see><c>&amp;</c>
///     If an <see ref="affinity_partitioner Class">affinity_partitioner</see> object is used, the reference must be a non-const l-value reference,
///     so that the algorithm can store state for future loops to re-use.
/// </param>
/// <remarks>
///     For more information, see <see cref="Parallel Algorithms"/>.
/// </remarks>
/**/
template <typename _Index_type, typename _Function, typename _Partitioner>
void parallel_for(_Index_type _First, _Index_type _Last, _Index_type _Step, const _Function& _Func, _Partitioner&& _Part)
{
    _Trace_ppl_function(PPLParallelForEventGuid, _TRACE_LEVEL_INFORMATION, CONCRT_EVENT_START);
    _Parallel_for_impl(_First, _Last, _Step, _Func, std::forward<_Partitioner>(_Part));
    _Trace_ppl_function(PPLParallelForEventGuid, _TRACE_LEVEL_INFORMATION, CONCRT_EVENT_END);
}

/// <summary>
///     <c>parallel_for</c> iterates over a range of indices and executes a user-supplied function at each iteration, in parallel.
/// </summary>
/// <typeparam name="_Index_type">
///     The type of the index being used for the iteration.
/// </typeparam>
/// <typeparam name="_Function">
///     The type of the function that will be executed at each iteration.
/// </typeparam>
/// <param name="_First">
///     The first index to be included in the iteration.
/// </param>
/// <param name="_Last">
///     The index one past the last index to be included in the iteration.
/// </param>
/// <param name="_Func">
///     The function to be executed at each iteration. This may be a lambda expression, a function pointer, or any object
///     that supports a version of the function call operator with the signature
///     <c>void operator()(</c><typeparamref name="_Index_type"/><c>)</c>.
/// </param>
/// <param name="_Part">
///     A reference to the partitioner object. The argument can be one of <c>const</c> <see ref="default_partitioner Class">default_partitioner</see><c>&amp;</c>,
///     <c>const</c> <see ref="fixed_partitioner Class">fixed_partitioner</see><c>&amp;</c> or <see ref="affinity_partitioner Class">affinity_partitioner</see><c>&amp;</c>
///     If an <see ref="affinity_partitioner Class">affinity_partitioner</see> object is used, the reference must be a non-const l-value reference,
///     so that the algorithm can store state for future loops to re-use.
/// </param>
/// <remarks>
///     For more information, see <see cref="Parallel Algorithms"/>.
/// </remarks>
/**/
template <typename _Index_type, typename _Function>
void parallel_for(_Index_type _First, _Index_type _Last, const _Function& _Func, const fixed_partitioner& _Part)
{
    parallel_for(_First, _Last, _Index_type(1), _Func, _Part);
}

}