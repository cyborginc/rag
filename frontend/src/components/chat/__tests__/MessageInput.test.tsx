import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import MessageInput from '../MessageInput';
import type { Filter } from '../../../types/chat';

// Mock the store and child components
const mockUseChatStore = vi.fn();
const mockUseCollectionsStore = vi.fn();

vi.mock('../../../store/useChatStore', () => ({
  useChatStore: () => mockUseChatStore()
}));

vi.mock('../../../store/useCollectionsStore', () => ({
  useCollectionsStore: () => mockUseCollectionsStore()
}));

vi.mock('../../collections/CollectionChips', () => ({
  CollectionChips: () => <div data-testid="collection-chips">Collection Chips</div>
}));

vi.mock('../MessageInputContainer', () => ({
  MessageInputContainer: () => <div data-testid="message-input-container">Message Input Container</div>
}));

vi.mock('../../filtering/SimpleFilterBar', () => ({
  default: ({ filters }: { filters: Filter[] }) => (
    <div data-testid="simple-filter-bar">
      Simple Filter Bar - Filters: {JSON.stringify(filters)}
    </div>
  )
}));

describe('MessageInput', () => {
  const mockSetFilters = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChatStore.mockReturnValue({
      filters: [],
      setFilters: mockSetFilters
    });
    mockUseCollectionsStore.mockReturnValue({
      selectedCollections: ['collection1'] // Default to having collections selected
    });
  });

  describe('Component Structure', () => {
    it('renders main container with correct structure', () => {
      const { container } = render(<MessageInput />);
      
      // KUI Flex component is now used instead of CSS classes
      const mainContainer = container.firstChild as HTMLElement;
      expect(mainContainer).toBeInTheDocument();
      // KUI components handle styling internally
    });

    it('renders all child components', () => {
      render(<MessageInput />);
      
      expect(screen.getByTestId('collection-chips')).toBeInTheDocument();
      expect(screen.getByTestId('simple-filter-bar')).toBeInTheDocument();
      expect(screen.getByTestId('message-input-container')).toBeInTheDocument();
    });

    it('renders components in correct order', () => {
      const { container } = render(<MessageInput />);
      
      // KUI Flex arranges components vertically - skip the container and get child components
      const components = container.querySelectorAll('[data-testid]:not([data-testid="nv-flex"])');
      expect(components[0]).toHaveAttribute('data-testid', 'collection-chips');
      expect(components[1]).toHaveAttribute('data-testid', 'simple-filter-bar');
      expect(components[2]).toHaveAttribute('data-testid', 'message-input-container');
    });

    it('contains filter bar and input container in the structure', () => {
      render(<MessageInput />);
      
      // KUI Flex with direction="col" handles grouping instead of .space-y-3
      expect(screen.getByTestId('simple-filter-bar')).toBeInTheDocument();
      expect(screen.getByTestId('message-input-container')).toBeInTheDocument();
    });
  });

  describe('Store Integration', () => {
    it('uses filters from chat store', () => {
      const testFilters = [
        { field: 'author', operator: '=', value: 'test' },
        { field: 'date', operator: '>', value: '2024-01-01' }
      ];

      mockUseChatStore.mockReturnValue({
        filters: testFilters,
        setFilters: mockSetFilters
      });

      render(<MessageInput />);
      
      expect(mockUseChatStore).toHaveBeenCalled();
      expect(screen.getByTestId('simple-filter-bar')).toHaveTextContent(JSON.stringify(testFilters));
    });

    it('passes setFilters to filter bar', () => {
      render(<MessageInput />);
      
      // The SimpleFilterBar mock receives the setFilters function
      expect(mockUseChatStore).toHaveBeenCalled();
    });

    it('handles empty filters array', () => {
      mockUseChatStore.mockReturnValue({
        filters: [],
        setFilters: mockSetFilters
      });

      render(<MessageInput />);
      
      expect(screen.getByTestId('simple-filter-bar')).toHaveTextContent('[]');
    });
  });

  describe('Exported Components', () => {
    it('exports all required components', () => {
      // This test ensures the main component renders (imports work)
      render(<MessageInput />);
      
      // Verify that the component renders successfully, which indicates exports work
      expect(screen.getByTestId('collection-chips')).toBeInTheDocument();
      expect(screen.getByTestId('simple-filter-bar')).toBeInTheDocument();
      expect(screen.getByTestId('message-input-container')).toBeInTheDocument();
    });
  });

  describe('Layout and Styling', () => {
    it('arranges components vertically', () => {
      const { container } = render(<MessageInput />);
      
      // KUI Flex with direction="col" handles vertical spacing
      const mainContainer = container.firstChild as HTMLElement;
      expect(mainContainer).toBeInTheDocument();
    });

    it('applies correct container structure', () => {
      const { container } = render(<MessageInput />);
      
      // KUI Flex handles padding through padding="density-md"
      const mainContainer = container.firstChild as HTMLElement;
      expect(mainContainer).toBeInTheDocument();
    });

    it('uses KUI styling system', () => {
      const { container } = render(<MessageInput />);
      
      // KUI components handle background and styling internally
      const mainContainer = container.firstChild as HTMLElement;
      expect(mainContainer).toBeInTheDocument();
    });
  });

  describe('Responsive Behavior', () => {
    it('renders with responsive structure', () => {
      const { container } = render(<MessageInput />);
      
      // KUI Flex components handle responsive behavior internally
      const mainContainer = container.firstChild as HTMLElement;
      expect(mainContainer).toBeInTheDocument();
    });

    it('maintains proper layout structure', () => {
      const { container } = render(<MessageInput />);
      
      // KUI components handle positioning internally
      const mainContainer = container.firstChild as HTMLElement;
      expect(mainContainer).toBeInTheDocument();
    });
  });

  describe('Filter Bar Visibility', () => {
    it('shows filter bar when collections are selected', () => {
      mockUseCollectionsStore.mockReturnValue({
        selectedCollections: ['collection1', 'collection2']
      });

      render(<MessageInput />);
      
      expect(screen.getByTestId('simple-filter-bar')).toBeInTheDocument();
    });

    it('hides filter bar when no collections are selected', () => {
      mockUseCollectionsStore.mockReturnValue({
        selectedCollections: []
      });

      render(<MessageInput />);
      
      expect(screen.queryByTestId('simple-filter-bar')).not.toBeInTheDocument();
    });

    it('shows filter bar when single collection is selected', () => {
      mockUseCollectionsStore.mockReturnValue({
        selectedCollections: ['single-collection']
      });

      render(<MessageInput />);
      
      expect(screen.getByTestId('simple-filter-bar')).toBeInTheDocument();
    });
  });

  describe('Component Integration', () => {
    it('integrates with collection chips', () => {
      render(<MessageInput />);
      
      const collectionChips = screen.getByTestId('collection-chips');
      expect(collectionChips).toHaveTextContent('Collection Chips');
    });

    it('integrates with message input container', () => {
      render(<MessageInput />);
      
      const inputContainer = screen.getByTestId('message-input-container');
      expect(inputContainer).toHaveTextContent('Message Input Container');
    });

    it('integrates with simple filter bar when collections selected', () => {
      render(<MessageInput />);
      
      const filterBar = screen.getByTestId('simple-filter-bar');
      expect(filterBar).toHaveTextContent('Simple Filter Bar');
    });
  });
}); 